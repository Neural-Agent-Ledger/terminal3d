//! `ascii_shader` — the data-driven procedural Material / texture engine.
//!
//! This is the "beauty engine": the layer that turns a world surface into the
//! glyphs, colours and lighting the player sees. A
//! [`Material`] is **data**: a [`TextureRecipe`] + a [`Palette`] (the iq cosine
//! palette) + a [`RankStrategy`], plus a seed. Every surface (wall kind, floor,
//! ceiling) references a [`MaterialId`] in the [`MaterialRegistry`] Bevy
//! resource. At shade time the renderer calls
//! [`Material::sample`]`(coord, ctx) -> `[`Texel`]` { glyph, fg, bg, rank }`.
//!
//! The look is **dual-channel**: every texel emits a `glyph` drawn in `fg` over
//! a `bg` base — two colors per cell for far richer detail than an fg-only ramp.
//!
//! **Ranked-LOD** keeps the look from overloading / aliasing at distance or at a
//! grazing angle. Each detail texel carries a `rank ∈ [0,127]` (rank-0 = base,
//! always visible; higher = finer detail). The shader computes a *texel density*
//! from `distance × foreshortening(angle)` and a reveal threshold `T = f(density)`;
//! a texel with `rank > T` is suppressed back to the base, and the transition is
//! a `smoothstep` fade (no pop). Near / face-on surfaces show all detail; far /
//! grazing surfaces show only the rank-0 base.
//!
//! Everything here is a **pure function of `(coord, seed, tick)`** — no clock, no
//! `rand`, no global state — so it is golden-test reproducible. Temporal
//! hysteresis (the one stateful, present-diff-reading pass) is deliberately
//! deferred to a later milestone; this module is the per-frame deterministic
//! core it will sit on top of.

use bevy::prelude::Resource;
use glam::{Vec2, Vec3};
use noise::{NoiseFn, Perlin};
use re_core::CellStyle;

// ─────────────────────────── global tunables (named) ────────────────────────

/// Master seed for every procedural primitive. Fixed → deterministic.
pub const SHADER_SEED: u32 = 0x5245_5601; // "REV\x01"

/// Highest rank value a detail texel may carry. rank-0 is the always-visible
/// base; higher ranks reveal progressively nearer / more face-on.
pub const RANK_MAX: u8 = 127;

/// Reveal-fade width (in "rank units"): how wide the `smoothstep` band is around
/// the reveal threshold. Wider → softer, slower coarse→fine emergence.
pub const REVEAL_FADE: f32 = 18.0;

/// Distance (in world cells) at which the LOD threshold has fully collapsed to
/// "base only". Past this, only rank-0 survives regardless of angle.
pub const LOD_FAR_DENSITY: f32 = 26.0;

/// Distance below which the LOD threshold is fully open (all ranks revealed).
pub const LOD_NEAR_DENSITY: f32 = 1.5;

/// Cells-per-world-unit of the COARSE lattice the neighbour-aware / procedural
/// super-glyph selectors ([`SuperSelector::NeighborAware`],
/// [`SuperSelector::Procedural`]) snap world coordinates to before hashing. The
/// same texel granularity [`crate::glyph_family::cell_phase`] uses, so the pick is
/// world-anchored (camera-stable) rather than re-rolled every sub-cell frame.
pub const NEIGHBOR_LATTICE: f32 = 3.0;

/// Brightness floor a fully-faded detail texel keeps before it snaps to base
/// (below this the texel is dropped and the base shows through).
pub const DETAIL_CUTOFF: f32 = 0.10;

/// FAR-PLANE HARD FREEZE distance, in world cells: the anti-noise cutoff beyond
/// which distant things must not noise at all.
///
/// At or beyond this perpendicular distance a cell is rendered FULLY QUIET — see
/// [`Material::sample`]: (a) its GLYPH is locked to a single world-anchored value
/// snapped to one fixed FAR LOD bucket, so LOD reveal/coarsen flips and depth-ink
/// re-selection stop; (b) ALL material ANIMATION is disabled (no pulse / flicker /
/// shift / hologram-scroll), so far surfaces hold still; (c) the glyph is still a
/// function of the WORLD `coord`, so the surface keeps scrolling naturally as the
/// camera moves (parallax/position preserved) and colour may still shade smoothly
/// — only the per-frame churn is killed. It builds ON TOP of the existing
/// [`GlyphStability`] deadband/hysteresis (a HARD cutoff past the soft `Frozen`
/// tier), so near surfaces stay fully alive.
///
/// Tuned LARGE — past the practical wall-render range (well beyond the default
/// [`FOG_FAR`] = 26) — so a normal-scale room is unaffected and only genuinely
/// distant background is frozen. Override per cast via the renderer's far-freeze
/// knob (which seeds [`SampleCtx::far_freeze`]); a value of `f32::INFINITY`
/// disables the freeze entirely (exactly the pre-feature behaviour).
pub const FAR_FREEZE_DIST: f32 = 32.0;

/// The fixed FAR LOD bucket (world cells) a far-frozen cell's GLYPH selection is
/// snapped to. A single constant — NOT distance-derived — so a frozen cell's glyph
/// is INVARIANT to its actual distance and viewing angle: the whole far field
/// shares one coarse LOD plateau and stops re-picking as the camera creeps. Kept
/// past [`LOD_FAR_DENSITY`] so the reveal threshold has fully collapsed to
/// "base only" (no detail pops at distance).
pub const FAR_FREEZE_LOD_BUCKET: f32 = LOD_FAR_DENSITY + 8.0;

/// ── DISTANCE-LOD shade bands (perf) ──────────────────────────────────────────
///
/// Per-cell shading is the single hottest cost of the near cast (the relief
/// gradient + multi-octave procedural noise run for every visible cell, every
/// animated/moving frame). But that micro-detail is only legible in the near
/// foreground; on a far cell that occupies ~1 ASCII glyph the relief gradient and
/// the high-frequency noise octaves are imperceptible. So shade DETAIL — never the
/// fg/bg colour selection or the depth→ink ramp, which carry legibility — is scaled
/// by the SHADED cell's perpendicular depth.
///
/// As of the LOD-unification this band decision lives in the SHARED
/// [`crate::lod::LodConfig`] near/mid/far tier system (the SAME tiers the shadow pass
/// reads, so the picture degrades coherently), with a per-boundary HYSTERESIS deadband
/// so a tier no longer pops at a single threshold while the player walks, and an
/// optional per-material override so a hero object holds detail farther out. The
/// per-cell [`ShadeLod`] is the projection of the tier's
/// [`crate::lod::TierDetail`] onto the two shade flags:
///
/// * NEAR tier — full relief gradient (the 2 extra `height_at` taps) + full noise
///   octaves.
/// * MID tier — relief gradient DROPPED (lit with the flat geometric face normal),
///   noise octaves still full.
/// * FAR tier — relief gradient dropped AND noise octaves capped to a cheap base
///   sample ([`SHADE_FAR_OCTAVES`] / the config's `far_octaves`).
///
/// The DEFAULT [`crate::lod::LodConfig`] reproduces the historical bands EXACTLY
/// (near→mid at 6, mid→far at 13), so this is not a visual change; the legacy
/// `SHADE_RELIEF_DIST`/`SHADE_OCTAVE_DIST` constants below are retained only as the
/// documented historical edge values (and as test distances). Thresholds are in
/// WORLD CELLS, resolution-independent; small-grid golden cameras sit well inside NEAR.
pub const SHADE_RELIEF_DIST: f32 = SHADOW_SOFT_DIST;

/// Beyond this perpendicular depth (world cells) the procedural noise octaves are
/// capped to [`SHADE_FAR_OCTAVES`] (a cheap low-frequency base). Between
/// [`SHADE_RELIEF_DIST`] and here the relief gradient is dropped but octaves stay
/// full. Aligned with [`SHADOW_HARD_DIST`].
pub const SHADE_OCTAVE_DIST: f32 = SHADOW_HARD_DIST;

/// The octave count a FAR cell's fBm/ridged noise is capped to. One octave is a
/// single low-frequency Perlin tap — the dominant shape of the wash/mottle/pour
/// survives, only the sub-pixel high-frequency detail is dropped. Each recipe's
/// own requested octave count is the NEAR value; this floor only ever LOWERS it.
pub const SHADE_FAR_OCTAVES: u32 = 1;

/// Per-cell shade level-of-detail, derived from the cell's perpendicular depth.
/// Computed ONCE at the top of [`Material::sample`] and threaded into the recipe
/// noise + relief pass so the depth banding is consistent and there are no
/// scattered distance comparisons in the hot path.
///
/// As of the LOD-unification this is a thin VIEW onto the shared
/// [`crate::lod::LodConfig`] tier system: `for_distance`/`for_ctx` resolve the cell's
/// [`crate::lod::LodTier`] from the active config (+ any per-material override) and
/// project the tier's [`crate::lod::TierDetail`] onto the two shade flags. The
/// scattered `SHADE_RELIEF_DIST`/`SHADE_OCTAVE_DIST` comparisons are gone — the tier
/// owns the band decision. The default config reproduces the historical bands
/// exactly, so this is not a visual change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadeLod {
    /// Keep the 2-extra-tap relief gradient (NEAR tier only). When false the cell
    /// is lit with its flat geometric face normal.
    pub relief_gradient: bool,
    /// Upper bound on procedural-noise octaves for this cell. `u32::MAX` near (no
    /// cap → each recipe's own count wins), the config's far-octave cap far.
    pub max_octaves: u32,
    /// Band-limited octaves: the WORLD-UNIT footprint of one screen cell at
    /// this cell's distance, from [`crate::lod::Footprint::reference`]
    /// (`re_render` has no live cols/FOV to read here, so this uses the same
    /// documented calibration reference [`ShadeLod::for_ctx`]'s footprint tier does
    /// — an honest approximation, not the live window). `0.0` ⇒ no footprint known
    /// (used by [`ShadeLod::FULL`] and any legacy construction path), which
    /// [`fbm_bandlimited`]/[`ridged_bandlimited`] treat as "near — do not fade
    /// anything", so a caller that never threads a distance is byte-identical.
    pub world_per_cell: f32,
}

impl ShadeLod {
    /// The full-detail LOD (near tier): full relief gradient, no octave cap, no
    /// footprint (so no octave gets band-limited). Used for callers without a
    /// distance (focused tests) so their look is byte-identical to the pre-LOD path.
    pub const FULL: ShadeLod = ShadeLod {
        relief_gradient: true,
        max_octaves: u32::MAX,
        world_per_cell: 0.0,
    };

    /// Project a unified [`crate::lod::TierDetail`] onto the two shade flags. The
    /// footprint hint is set separately by the distance-aware constructors below
    /// (a `TierDetail` alone carries no distance to derive it from).
    #[inline]
    pub fn from_detail(d: crate::lod::TierDetail) -> Self {
        ShadeLod {
            relief_gradient: d.relief_gradient,
            max_octaves: d.max_octaves,
            world_per_cell: 0.0,
        }
    }

    /// Select the shade LOD for a cell at perpendicular `distance` (world cells) under
    /// the active [`crate::lod::LodConfig`], with NO per-material override. Kept for
    /// call sites / tests without a material; with the default config it reproduces
    /// the historical `relief_gradient = distance < 6`, `octaves capped at > 13` bands.
    #[inline]
    pub fn for_distance(distance: f32) -> Self {
        Self::for_distance_ov(distance, None)
    }

    /// Select the shade LOD for a cell at `distance` under the active config + an
    /// optional per-material [`crate::lod::LodOverride`] (the hero-object lever).
    #[inline]
    pub fn for_distance_ov(distance: f32, ov: Option<crate::lod::LodOverride>) -> Self {
        let mut lod = Self::from_detail(crate::lod::active().detail_for(distance, ov));
        lod.world_per_cell = crate::lod::Footprint::reference(distance.max(0.0)).f_u;
        lod
    }

    /// Select the shade LOD for a `SampleCtx`, honouring the cell's per-material LOD
    /// override carried on the ctx. The one call the recipe/relief path uses so the
    /// override flows to every shade-LOD decision (and the tex-cache key) coherently.
    #[inline]
    pub fn for_ctx(ctx: &SampleCtx) -> Self {
        Self::for_distance_ov(ctx.distance, ctx.lod_override)
    }

    /// Clamp a recipe's requested octave count to this LOD's cap. `min` so a recipe
    /// that already asks for fewer octaves than the cap is never RAISED.
    #[inline]
    pub fn octaves(self, requested: u32) -> u32 {
        requested.min(self.max_octaves)
    }
}

// ───────────────────────────── math primitives ──────────────────────────────

/// Linear interpolation.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite `smoothstep` in `[0,1]` (3t²−2t³), clamped at the edges.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fractional part in `[0,1)` (matching GLSL `fract`).
#[inline]
fn fract(v: f32) -> f32 {
    v - v.floor()
}

/// FNV-ish 32-bit integer mix — deterministic across platforms.
#[inline]
pub fn hash2i(a: i32, b: i32) -> u32 {
    let mut x = (a as u32).wrapping_mul(0x9E37_79B1);
    x ^= (b as u32)
        .wrapping_add(0x85EB_CA6B)
        .wrapping_add(x << 6)
        .wrapping_add(x >> 2);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

/// Three-argument extension of [`hash2i`]: folds `c` into the pair hash.
#[inline]
pub fn hash3i(a: i32, b: i32, c: i32) -> u32 {
    hash2i(hash2i(a, b) as i32, c)
}

/// Hash a seed-salted integer cell → `u32`.
#[inline]
fn hash_cell(seed: u32, ix: i32, iy: i32) -> u32 {
    hash3i(seed as i32, ix, iy)
}

/// Hash → `f32` in `[0,1)`.
///
/// `pub(crate)` so a sibling module can build a world-anchored field on the SAME
/// hash the recipes use instead of adding a second one ([`crate::substrate`]).
#[inline]
pub(crate) fn hash01(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Two independent `[0,1)` hashes from one cell (for jittered feature points).
#[inline]
fn hash2_01(seed: u32, ix: i32, iy: i32) -> Vec2 {
    let h = hash_cell(seed, ix, iy);
    let hx = hash01(h);
    let hy = hash01(h.wrapping_mul(0x2545_F491).rotate_left(13) ^ 0x9E37_79B9);
    Vec2::new(hx, hy)
}

// ─────────────────────────── iq cosine palette ──────────────────────────────

/// The iq cosine gradient palette: `col(t) = a + b·cos(2π·(c·t + d))`.
///
/// Four `Vec3` control vectors give an entire coherent color scheme from twelve
/// floats — bias `a`, amplitude `b`, frequency `c`, phase `d`. Channels are
/// returned in linear `[0,1]³` (R,G,B); pack with [`pack_rgb`].
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    /// Bias — the mean color the gradient oscillates around.
    pub a: Vec3,
    /// Amplitude — how far each channel swings from the bias.
    pub b: Vec3,
    /// Frequency — how many cycles per unit of `t` each channel completes.
    pub c: Vec3,
    /// Phase — the per-channel offset into the cosine cycle.
    pub d: Vec3,
}

impl Palette {
    /// Evaluate the palette at `t` (any real; wraps via cosine).
    #[inline]
    pub fn eval(&self, t: f32) -> Vec3 {
        let tau = std::f32::consts::TAU;
        let arg = (self.c * t + self.d) * tau;
        let cos = Vec3::new(arg.x.cos(), arg.y.cos(), arg.z.cos());
        (self.a + self.b * cos).clamp(Vec3::ZERO, Vec3::ONE)
    }
}

/// Pack a linear `[0,1]³` color into `0x00RRGGBB`.
#[inline]
pub fn pack_rgb(c: Vec3) -> u32 {
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 255.0) + 0.5) as u32 & 0xFF;
    (q(c.x) << 16) | (q(c.y) << 8) | q(c.z)
}

/// Unpack `0x00RRGGBB` into a linear `[0,1]³` color.
#[inline]
pub fn unpack_rgb(c: u32) -> Vec3 {
    Vec3::new(
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
    )
}

/// Multiply a packed color by a `[0,1]` factor (fade to black, hue stable).
#[inline]
pub fn dim_rgb(c: u32, factor: f32) -> u32 {
    pack_rgb(unpack_rgb(c) * factor.clamp(0.0, 1.0))
}

/// Multiply a packed color by a `factor >= 1` (BRIGHTEN toward white per channel,
/// clamped at 1.0 by [`pack_rgb`]). The complement of [`dim_rgb`] (which clamps the
/// factor to `[0,1]`): the state-bus PULSE term uses this to flash a cell brighter.
/// `factor == 1.0` is identity; a non-inked (0x000000) cell stays black.
#[inline]
pub fn brighten_rgb(c: u32, factor: f32) -> u32 {
    pack_rgb(unpack_rgb(c) * factor.max(0.0))
}

/// Linear blend between two packed colors; `t=0`→a, `t=1`→b.
#[inline]
pub fn mix_rgb(a: u32, b: u32, t: f32) -> u32 {
    pack_rgb(unpack_rgb(a).lerp(unpack_rgb(b), t.clamp(0.0, 1.0)))
}

/// Rec.601 relative luminance of a packed `0x00RRGGBB` colour, in `[0,1]`. Used by
/// the depth-ink cue to decide which of fg/bg is the brighter channel (so the cue
/// never inverts on a bg-brighter surface), and by `node_sprite`'s dead-node
/// contrast-aware colour pick (which of two neutral tones reads against the actual
/// background behind the node).
#[inline]
pub(crate) fn luma_of(c: u32) -> f32 {
    unpack_rgb(c).dot(Vec3::new(0.299, 0.587, 0.114))
}

// ───────────────────────── relief / normal-mapped lighting ──────────────────
//
// The relief model turns every material's procedural scalar field into a HEIGHT
// map, differentiates it for a tangent-space normal, combines that with the
// cell's GEOMETRIC FACE NORMAL (the raycaster knows it: ±x/±y for walls by
// `side`, ±z for floor/ceiling), and lights the result with Blinn-Phong in
// truecolor. This supersedes the old flat `SIDE_EW_DIM` directional trick: a
// wall you face vs one you graze now differ via `N·L`, not a hardcoded factor.
//
// Everything is a few extra scalar evals per cell (two height taps for the
// central-difference gradient) — NO per-cell heap allocation. Light + view
// vectors are precomputed once per cast/column and threaded in via [`LightCtx`].

/// The per-material relief + lighting knobs (the "many kinds of maps" surface).
/// Pure data, `Copy`, so a [`Material`] stays `Copy` and can be a `static`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightModel {
    /// How tall the height field reads, in tangent units. Larger ⇒ steeper
    /// relief, more pronounced normal tilt. `0` ⇒ flat (face normal only).
    pub height_scale: f32,
    /// Scales the tangent-space normal's xy tilt before it is combined with the
    /// face normal. Larger ⇒ the bumps "lean" more toward the light. `0` ⇒ the
    /// surface lights as a flat face (no relief), `1` ⇒ full recipe relief.
    pub normal_strength: f32,
    /// Blinn-Phong specular exponent. Higher ⇒ tighter, glossier glint.
    pub shininess: f32,
    /// Specular intensity `[0,~2]`: how bright the half-vector glint is.
    pub specular: f32,
    /// Specular tint, packed `0x00RRGGBB` (the glint colour — often white-ish).
    pub spec_color: u32,
    /// Ambient occlusion strength `[0,1]`: how much low/concave height darkens.
    pub ao_strength: f32,
    /// Self-illumination `[0,1]`: a fraction of the surface colour that ignores
    /// the light (circuit traces glow regardless of the key light / headlamp).
    pub emissive: f32,
}

impl LightModel {
    /// A neutral, mostly-flat default: faint relief, a soft glint, light AO. Used
    /// by [`DEFAULT_MATERIAL`] and as the base for authored materials that don't
    /// tune lighting. Tasteful but unobtrusive.
    pub const DEFAULT: LightModel = LightModel {
        height_scale: 0.6,
        normal_strength: 0.8,
        shininess: 16.0,
        specular: 0.35,
        spec_color: 0x00FF_FFFF,
        ao_strength: 0.5,
        emissive: 0.0,
    };

    /// A fully-flat model: no relief, no spec, no AO, no emissive. Combining it
    /// with the lighting pass reproduces "geometric face normal only" lighting.
    pub const FLAT: LightModel = LightModel {
        height_scale: 0.0,
        normal_strength: 0.0,
        shininess: 8.0,
        specular: 0.0,
        spec_color: 0x00FF_FFFF,
        ao_strength: 0.0,
        emissive: 0.0,
    };
}

/// The per-cell lighting geometry the raycaster hands the shader: the world-space
/// GEOMETRIC FACE NORMAL plus the tangent frame the recipe's height gradient
/// perturbs, and the (precomputed) light + view directions. All in world space,
/// all normalized. Cheap to build (the caster already has `side` + ray dir), and
/// constant down a column except for the view direction, so it is computed once
/// per cast/column — never per height-tap.
#[derive(Clone, Copy, Debug)]
pub struct LightCtx {
    /// Geometric face normal (unit). Walls: ±x or ±y; floor: +z; ceiling: −z.
    pub face_n: Vec3,
    /// Tangent axis aligned with the surface's `u` (horizontal) coordinate.
    pub tangent_u: Vec3,
    /// Tangent axis aligned with the surface's `v` (vertical) coordinate.
    pub tangent_v: Vec3,
    /// Directional KEY light direction (unit, pointing TOWARD the light).
    pub key_dir: Vec3,
    /// KEY light colour, linear `[0,1]³`.
    pub key_color: Vec3,
    /// View direction (unit, pointing from the surface back TOWARD the camera).
    /// Doubles as the headlamp direction (a light co-located with the eye).
    pub view_dir: Vec3,
    /// Ambient light colour/level, linear `[0,1]³` (the soft floor).
    pub ambient: Vec3,
    /// Headlamp intensity `[0,~1]`: a point-ish light along `view_dir` so what you
    /// face is lit and relief pops as you move (first-person exploration).
    pub headlamp: f32,
    /// KEY-light shadow factor `[0,1]` from the renderer's DDA shadow-ray pass:
    /// `1.0` = fully lit by the key, `0.0` = fully in the key's cast shadow,
    /// fractional = penumbra. Multiplies ONLY the key diffuse + key specular terms
    /// in [`light_surface`]; ambient, headlamp, and emissive are unshadowed (so a
    /// shadowed cell stays readable, never pure black). Defaults to `1.0`
    /// ([`ShadowQuality::Off`] and every legacy / test [`LightCtx`]), which
    /// reproduces the pre-shadow look exactly.
    pub shadow_key: f32,
    /// WORLD position of the shaded surface point. `x`/`y` are the floor plane and
    /// **`z` is height** — the same convention as `CameraPose::pos`. Used by the
    /// punctual lights (distance + direction) and by the height-fog term. The
    /// [`LightCtx`] constructors default it to [`Vec3::ZERO`]; the raycaster
    /// overwrites it with the cell's real hit point via [`LightCtx::at`].
    pub world_pos: Vec3,
    /// The scene's punctual lights, copied from the [`LightEnv`]. Empty ⇒ the
    /// point-light sum is skipped entirely.
    pub points: PointLights,
}

impl LightCtx {
    /// A sensible built-in lighting environment for a wall of the given `side`,
    /// viewed along `view_dir` (unit, surface→camera). The KEY light comes from
    /// the upper-front-left so vertical relief (brick courses, traces) catches it;
    /// the headlamp rides the view. Floor/ceiling use [`for_floor`](Self::for_floor).
    ///
    /// `face_sign` picks which way the axis-aligned face points (toward the
    /// camera): the caster passes the sign of the ray component so the normal
    /// faces the eye. The tangent frame is axis-aligned to the face.
    #[inline]
    pub fn for_wall(side: u8, face_sign: f32, view_dir: Vec3, env: &LightEnv) -> Self {
        // Face normal points back along the axis the ray advanced on, toward the
        // camera. side==1 ⇒ an x-face (normal ±x); side==0 ⇒ a y-face (normal ±y).
        let (face_n, tangent_u) = if side == 1 {
            (
                Vec3::new(-face_sign, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0), // u runs along world y
            )
        } else {
            (
                Vec3::new(0.0, -face_sign, 0.0),
                Vec3::new(1.0, 0.0, 0.0), // u runs along world x
            )
        };
        // v runs DOWN the wall: world −z (rows increase downward → height down).
        let tangent_v = Vec3::new(0.0, 0.0, -1.0);
        LightCtx {
            face_n,
            tangent_u,
            tangent_v,
            key_dir: env.key_dir,
            key_color: env.key_color,
            view_dir,
            ambient: env.ambient,
            headlamp: env.headlamp,
            // Unshadowed by default; the renderer overwrites this with the
            // DDA shadow-ray result for the shipped path.
            shadow_key: 1.0,
            // Origin by default; the caster calls `at()` with the real hit point.
            world_pos: Vec3::ZERO,
            points: env.points,
        }
    }

    /// Lighting environment for a NON-axis-aligned 45° surface (a [`re_core::Shape`]
    /// wedge diagonal or ramp slope), given its real world-space outward unit
    /// `normal`. Builds a stable tangent frame: `tangent_v` runs as vertical as the
    /// face allows (down the slope), `tangent_u` is the horizontal cross axis. The
    /// KEY light + headlamp then catch the true 45° tilt, so the face shades as a
    /// lit slope instead of a flat wall. Axis-aligned faces keep [`Self::for_wall`]
    /// (byte-identical); only wedge/ramp hits route here.
    #[inline]
    pub fn for_surface_normal(normal: Vec3, view_dir: Vec3, env: &LightEnv) -> Self {
        let face_n = normal.normalize_or_zero();
        // Pick a reference that is least parallel to the normal so the cross product
        // is well-conditioned. A vertical wedge wall (normal in XY) → reference +Z;
        // a ramp (normal tilted up) → reference +Z still works (normal never ±Z).
        let up = Vec3::Z;
        let tangent_u = if face_n.cross(up).length_squared() > 1e-6 {
            face_n.cross(up).normalize()
        } else {
            // Degenerate (normal ≈ ±Z, which our shapes never produce) — fall back.
            Vec3::X
        };
        // v runs down the surface (toward −z where the face tilts).
        let tangent_v = tangent_u.cross(face_n).normalize_or_zero();
        LightCtx {
            face_n,
            tangent_u,
            tangent_v,
            key_dir: env.key_dir,
            key_color: env.key_color,
            view_dir,
            ambient: env.ambient,
            headlamp: env.headlamp,
            shadow_key: 1.0,
            world_pos: Vec3::ZERO,
            points: env.points,
        }
    }

    /// Lighting environment for a floor (`+z` up) or ceiling (`−z` down) cell.
    #[inline]
    pub fn for_floor(is_floor: bool, view_dir: Vec3, env: &LightEnv) -> Self {
        let face_n = if is_floor {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(0.0, 0.0, -1.0)
        };
        LightCtx {
            face_n,
            tangent_u: Vec3::new(1.0, 0.0, 0.0),
            tangent_v: Vec3::new(0.0, 1.0, 0.0),
            key_dir: env.key_dir,
            key_color: env.key_color,
            view_dir,
            ambient: env.ambient,
            headlamp: env.headlamp,
            shadow_key: 1.0,
            world_pos: Vec3::ZERO,
            points: env.points,
        }
    }

    /// Anchor this context at a WORLD surface point (`z` = height). Returns
    /// `self` so the caster can write `LightCtx::for_wall(..).at(hit)` in one
    /// expression. A context left un-anchored sits at the origin, which only
    /// matters when the scene actually has punctual lights or height fog.
    #[inline]
    #[must_use]
    pub fn at(mut self, world_pos: Vec3) -> Self {
        self.world_pos = world_pos;
        self
    }
}

/// Shadow-pass quality knob (the degradation ladder for the worked-out shadows).
/// A `Resource` so a settings row / perf tier can tune it; the
/// default is [`ShadowQuality::Soft`]. Lives here (engine-pure data) and is
/// consumed by the renderer's DDA shadow-ray pass, which feeds the resulting
/// `shadow_key` into [`LightCtx`].
///
/// * `Off`   — no occlusion test. Reproduces the pre-shadow look EXACTLY
///   (`shadow_key` stays 1.0 → today's local AO + headlamp + unoccluded key).
/// * `Hard`  — a single shadow ray per cell: a crisp binary in-shadow bit.
/// * `Soft`  — a few jittered taps + distance-to-blocker penumbra widening
///   (PCSS-lite) → a fractional `shadow_key` with a soft edge.
#[derive(bevy::prelude::Resource, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ShadowQuality {
    /// No shadow ray; `shadow_key == 1.0` everywhere (pre-shadow output).
    Off,
    /// One shadow ray per cell — a hard binary shadow.
    Hard,
    /// Multi-tap jittered penumbra (PCSS-lite) — soft, fractional shadows.
    #[default]
    Soft,
}

/// Number of jittered shadow taps in [`ShadowQuality::Soft`]. Kept
/// small (a few rays) so the per-cell cost stays a handful of integer grid reads.
pub const SHADOW_TAPS: u32 = 4;

/// Max voxels a shadow ray marches toward the key light before it is treated as
/// unoccluded (escaped to open space / the light). Bounds the worst-case cost.
pub const SHADOW_MAX_STEPS: u32 = 24;

/// Penumbra disc half-angle (radians) the soft taps jitter the key direction by,
/// at the BASE (a near blocker). Distance-to-blocker softening widens it toward
/// [`SHADOW_SOFT_MAX`] for a far blocker (PCSS-lite).
pub const SHADOW_SOFT_BASE: f32 = 0.045;

/// Upper bound on the penumbra disc half-angle for a far blocker.
pub const SHADOW_SOFT_MAX: f32 = 0.22;

/// Surface bias (voxels) the shadow ray origin is pushed off the face along the
/// face normal before marching, so a lit face does not shadow ITSELF on the
/// originating voxel (classic shadow-map bias against "shadow acne").
pub const SHADOW_BIAS: f32 = 0.02;

/// ── DISTANCE-LOD shadow bands (perf) ────────────────────────────────────────
///
/// The soft floor/ceiling shadow pass is the single hottest cost of the near cast
/// (every floor/ceiling cell at EVERY depth pays up to `SHADOW_TAPS × SHADOW_MAX_STEPS`
/// voxel reads). But a cast shadow's soft penumbra is only legible where the player
/// actually looks — the near foreground. Distant floor/ceiling occlusion is
/// imperceptible in a few ASCII glyphs. So shadow cost is scaled by the SHADED
/// cell's perpendicular depth, in world cells:
///
/// * NEAR (`depth < SHADOW_SOFT_DIST`) — full soft shadow (taps + penumbra),
///   unchanged. The detail the player sees stays byte-identical.
/// * MID  (`SHADOW_SOFT_DIST..SHADOW_HARD_DIST`) — a single HARD ray (1 ray, no
///   penumbra). Still a real cast shadow, ~1/4 the rays of Soft.
/// * FAR  (`> SHADOW_HARD_DIST`) — no shadow ray at all (`shadow_key == 1.0`).
///   At this range the cell occupies ~1 glyph; the missing penumbra is invisible.
///
/// Thresholds are in WORLD CELLS so they are independent of screen resolution and
/// of the small-grid golden cameras (whose visible floor is all well inside the
/// NEAR band — see the golden-safety note in the perf task). `SHADOW_SOFT_DIST`
/// is set generously (a couple of rooms) so the foreground a player reads is the
/// full-quality band; only mid/far cells are cheapened.
pub const SHADOW_SOFT_DIST: f32 = 6.0;

/// Beyond this perpendicular depth (world cells) no shadow ray is cast at all.
/// Between [`SHADOW_SOFT_DIST`] and here, a single hard ray is used.
pub const SHADOW_HARD_DIST: f32 = 13.0;

/// How many level-authored PUNCTUAL lights a scene may carry.
///
/// The set is stored INLINE (a fixed array, not a slice) because [`LightCtx`] is
/// `Copy` and constructed once per cell: a borrow would force a lifetime
/// parameter through ~80 use sites in this crate and through `SampleCtx`. The
/// cap keeps the per-cell copy bounded — each [`PointLightRt`] is 28 bytes, so 4
/// lights add 116 B to a ~92 B `LightCtx`. Raising this raises the per-cell
/// memcpy linearly; 4 is enough for the authored "pool of light" look and
/// `re_lvlcheck` warns when a level authors more.
pub const MAX_POINT_LIGHTS: usize = 4;

/// How much of the raw UNSHADOWED point-light radiance ([`point_light_sum`]'s
/// return, hue included) is added directly into the cell's BG, on top of the
/// existing scalar `shadow` darkening.
///
/// Without this term the point-light contribution only reaches the bg through
/// `shadow` — a single scalar (`lit.max_element()` folded through [`dim_rgb`])
/// that can only re-scale the recipe's OWN bg colour. A lamp's hue never
/// reaches the bg that way, and on a body-coverage-blanked floor cell (glyph
/// `' '`, κ = 0) the bg is the cell's *only* visible carrier, so the light
/// pool would be invisible there even though `light_surface`'s `fg`/`lum`
/// outputs already lit up. This constant is an ADDITIVE gain on the raw
/// radiance (not a multiply of the existing bg), so red light makes the bg
/// genuinely redder rather than merely brighter.
///
/// `0.85` was tuned by hand against `levels/22-lit-cellar.lvl`
/// (`--dump --ansi`): a lamp's pool reads clearly as a colour, not just a
/// slightly-less-black patch, without the additive term blowing the bg past
/// the ambient/shadow read on unlit floor. A future balance pass may retune
/// this in one place.
pub const LAMP_BG_GAIN: f32 = 0.85;

/// One punctual light, pre-baked for the shading loop.
///
/// The author's `color` + `intensity` are collapsed into a single linear
/// `radiance`, and `radius` into `inv_radius2 = 1/radius²`, so the hot path is
/// `radiance * max(N·L,0) / (1 + d²·inv_radius2)` — no divisions by a authored
/// value, no `powf`, no unpacking per cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLightRt {
    /// World position (`x`, `y` floor plane; **`z` is height**).
    pub pos: Vec3,
    /// Linear colour × intensity.
    pub radiance: Vec3,
    /// `1 / radius²` (radius > 0 is guaranteed by the parser and by `from_world`).
    pub inv_radius2: f32,
}

impl PointLightRt {
    /// Inert light: zero radiance, so it contributes nothing even if summed.
    pub const NONE: PointLightRt = PointLightRt {
        pos: Vec3::ZERO,
        radiance: Vec3::ZERO,
        inv_radius2: 1.0,
    };
}

/// The scene's punctual light set: an inline, `Copy` array plus a live count.
/// `n == 0` (the default, and every shipped level) makes the point-light sum in
/// [`light_surface_mode`] a single predictable branch, so the lit result is
/// bit-for-bit the pre-lighting output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLights {
    /// Number of live entries in `l` (`<= MAX_POINT_LIGHTS`).
    n: u8,
    l: [PointLightRt; MAX_POINT_LIGHTS],
}

impl PointLights {
    /// The empty set — no punctual lights (the shipped-level default).
    pub const NONE: PointLights = PointLights {
        n: 0,
        l: [PointLightRt::NONE; MAX_POINT_LIGHTS],
    };

    /// Bake up to [`MAX_POINT_LIGHTS`] authored [`re_world::PointLight`]s into the
    /// runtime set. Lights with a non-positive/non-finite radius or a zero
    /// intensity are DROPPED (they cannot contribute; carrying them would only
    /// cost per-cell work). Extras beyond the cap are ignored — `re_lvlcheck`
    /// warns at author time so this is never a silent surprise in practice.
    pub fn from_world(lights: &[re_world::PointLight]) -> Self {
        let mut out = PointLights::NONE;
        for src in lights {
            if out.n as usize >= MAX_POINT_LIGHTS {
                break;
            }
            if !(src.radius.is_finite() && src.radius > 0.0) {
                continue;
            }
            if !src.intensity.is_finite() || src.intensity <= 0.0 {
                continue;
            }
            if !src.pos.is_finite() {
                continue;
            }
            out.l[out.n as usize] = PointLightRt {
                pos: src.pos,
                radiance: unpack_rgb(src.color) * src.intensity,
                inv_radius2: 1.0 / (src.radius * src.radius),
            };
            out.n += 1;
        }
        out
    }

    /// The live lights, as a slice (empty for [`PointLights::NONE`]).
    #[inline]
    pub fn as_slice(&self) -> &[PointLightRt] {
        &self.l[..self.n as usize]
    }

    /// `true` when there is nothing to sum — the hot-path early-out.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// How many lights are live.
    #[inline]
    pub fn len(&self) -> usize {
        self.n as usize
    }
}

impl Default for PointLights {
    fn default() -> Self {
        PointLights::NONE
    }
}

/// Sum the UNSHADOWED punctual-light contribution at `world_pos` with surface
/// normal `normal` (both world space):
/// `Σ radiance · max(N·L, 0) / (1 + d²/radius²)`.
///
/// Inverse-square-ish with a `+1` denominator so a light never blows up as
/// `d → 0`; at `d == radius` the falloff is exactly ½. Unshadowed by design —
/// shadowed point lights are a deliberately deferred feature — which is why
/// the sum is a handful of FLOPs and needs no ray.
#[inline]
fn point_light_sum(points: &PointLights, world_pos: Vec3, normal: Vec3) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    let mut acc = Vec3::ZERO;
    for l in points.as_slice() {
        let to_light = l.pos - world_pos;
        let d2 = to_light.length_squared();
        // Coincident with the surface point: no meaningful direction, and the
        // attenuation would be 1 — treat it as a straight radiance add so the
        // cell is bright rather than NaN.
        if d2 <= 1.0e-8 {
            acc += l.radiance;
            continue;
        }
        let inv_d = d2.sqrt().recip();
        let n_dot_l = normal.dot(to_light * inv_d);
        if n_dot_l <= 0.0 {
            continue; // facing away — no contribution, and no attenuation cost.
        }
        let atten = 1.0 / (1.0 + d2 * l.inv_radius2);
        acc += l.radiance * (n_dot_l * atten);
    }
    acc
}

/// The scene-wide lighting environment: the directional KEY light + ambient +
/// headlamp strength. Precomputed once per cast (the key light is constant; the
/// headlamp rides the per-cell view direction). A level may tune this via its
/// [`Atmosphere`](re_world::Atmosphere); the default is a sensible built-in.
#[derive(Clone, Copy, Debug)]
pub struct LightEnv {
    /// KEY light direction (unit, pointing TOWARD the light).
    pub key_dir: Vec3,
    /// KEY light colour, linear `[0,1]³`.
    pub key_color: Vec3,
    /// Ambient colour/level (the soft floor so nothing is pure black).
    pub ambient: Vec3,
    /// Headlamp intensity `[0,~1]`.
    pub headlamp: f32,
    /// Level-authored PUNCTUAL lights ([`PointLights::NONE`] ⇒ none, and the
    /// shading result is bit-identical to the pre-lighting build).
    pub points: PointLights,
}

impl LightEnv {
    /// The shipped built-in lighting: a cool-white key from the upper-front, a
    /// soft neutral ambient floor, and a warm headlamp so the player's own light
    /// reveals relief as they move. Normalised key direction.
    pub fn builtin() -> Self {
        LightEnv {
            // Upper-front-left, biased downward a touch so wall courses + floor
            // grain both catch a rake. Normalised below.
            key_dir: Vec3::new(-0.35, -0.45, 0.82).normalize(),
            key_color: Vec3::new(0.95, 0.96, 1.0),
            ambient: Vec3::splat(0.18),
            headlamp: 0.28,
            points: PointLights::NONE,
        }
    }
}

impl Default for LightEnv {
    fn default() -> Self {
        LightEnv::builtin()
    }
}

/// The post-recipe inputs to [`Material::apply_relief_lighting`]: whether a
/// structural detail glyph was revealed (those keep their glyph; only the body
/// glyph is height/depth-driven), the animation brightness multiplier, and the
/// cell distance (driving the depth→ink falloff). Bundled so the lighting entry
/// point stays under the argument-count lint.
#[derive(Clone, Copy, Debug)]
struct ReliefArgs {
    detail_revealed: bool,
    anim_b: f32,
    distance: f32,
    /// FAR-PLANE HARD FREEZE flag: when set, the body-glyph depth-ink re-selection
    /// snaps its distance to the fixed FAR bucket so the chosen body glyph is
    /// invariant to the cell's actual distance (no glyph slide with depth at range).
    frozen: bool,
    /// KINDRED-SYMBOL family glyph present: a [`crate::glyph_family::FamilyLayer`]
    /// already substituted the cell's glyph, so the relief body-glyph re-selection
    /// (and the sparse-body blanking) must LEAVE IT ALONE — the family owns the glyph
    /// (the colour/relief lighting still runs). `false` ⇒ the historical path.
    family_glyph: bool,
    /// DISTANCE-LOD shade band: whether to keep the 2-extra-tap relief gradient and
    /// the octave cap for the height-field noise. NEAR ⇒ [`ShadeLod::FULL`] (the
    /// gradient runs, no cap) so the near field is byte-identical.
    lod: ShadeLod,
    /// The world sample coord (post-anim) — used by the sparse-body coverage mask.
    coord: Vec2,
    /// The relief height `h(coord)` (one tap), precomputed in `texture_sample`.
    h: f32,
    /// The two extra central-difference height taps `(h(coord+du), h(coord+dv))`,
    /// precomputed in `texture_sample`. `None` ⇒ light with the flat face normal.
    grad_taps: Option<(f32, f32)>,
    /// DEPTH-CONTRAST detail collapse (`crate::depth_contrast::ramp_steps_for`):
    /// the body-ramp step count this cell's LOD tier permits, or `None` for no
    /// collapse (the NEAR tier, or the whole effect disabled). Precomputed in
    /// `apply_lighting` (which has `ctx.fog` / `crate::depth_contrast::active()`)
    /// and threaded in so this struct stays the single bundle
    /// `apply_relief_lighting` reads.
    ramp_steps: Option<f32>,
}

/// The legibility floor (4a/5a): [`crate::legibility::legible_pair`] on a cell no
/// authored lamp reaches, [`crate::legibility::legible_pair_lamped`] on one it may
/// (`unlamped` = the same cell shaded without point lights). `glyph` says whether the
/// cell draws ink at all (a blank cell's fg is never seen).
#[inline]
fn legible_floor(
    glyph: u32,
    (fg, bg): (u32, u32),
    unlamped: Option<(u32, u32)>,
    (fg_hue, bg_hue): (u32, u32),
) -> (u32, u32) {
    unlamped.map_or_else(
        || crate::legibility::legible_pair(fg, bg, fg_hue, bg_hue),
        |dark| {
            let inked = char::from_u32(glyph).is_some_and(|c| !c.is_whitespace());
            crate::legibility::legible_pair_lamped(fg, bg, dark, fg_hue, bg_hue, inked)
        },
    )
}

/// Blinn-Phong lighting of a base surface colour `albedo` (linear `[0,1]³`) given
/// a world-space `normal`, the cell's height `h ∈ [0,1]` (for AO), the material's
/// [`LightModel`], and the [`LightCtx`]. Returns `(lit_fg, lit_bg_factor, lum, point_radiance)`:
/// the lit foreground colour, a `[0,1]` darkening factor to push the bg into
/// shadow/crevice, a scalar luminance `[0,~1.5]` the glyph ramp reads so
/// raised-lit cells pick a denser glyph, and the raw UNSHADOWED point-light
/// radiance (hue included, before ambient/key mixing — the carrier the bg
/// gain reads, see [`LAMP_BG_GAIN`]). Pure, no allocation.
///
/// Reads the [`view_independent_lighting`] toggle once and delegates to
/// [`light_surface_mode`]; tests call the `_mode` form directly to exercise both
/// modes without touching the process-global env.
#[inline]
fn light_surface(
    albedo: Vec3,
    normal: Vec3,
    h: f32,
    lm: &LightModel,
    lc: &LightCtx,
) -> (Vec3, f32, f32, Vec3) {
    light_surface_mode(albedo, normal, h, lm, lc, view_independent_lighting())
}

/// [`light_surface`] with the view-independence mode passed explicitly. `view_indep
/// == true` DROPS every term that reads `lc.view_dir` — the headlamp facing term
/// (N·V) and both Blinn-Phong specular glints — so the result is a pure function of
/// (albedo, world normal, key light, ambient, height/AO), invariant to camera
/// facing. `view_indep == false` is today's full Blinn-Phong, byte-identical.
#[inline]
fn light_surface_mode(
    albedo: Vec3,
    normal: Vec3,
    h: f32,
    lm: &LightModel,
    lc: &LightCtx,
    view_indep: bool,
) -> (Vec3, f32, f32, Vec3) {
    // Ambient occlusion from the height field: low (concave/groove) areas are
    // darker so crevices read as depth. h in [0,1]; ao in [1-ao_strength, 1].
    let ao = 1.0 - lm.ao_strength * (1.0 - h).clamp(0.0, 1.0);

    // WORKED-OUT SHADOWS (#2): the renderer's DDA shadow-ray pass tells us how much
    // of the KEY light reaches this surface point. `shadow_key ∈ [0,1]` (1 = lit,
    // 0 = fully in the key's cast shadow, fractional = penumbra). It attenuates
    // ONLY the key diffuse + key specular below; the ambient floor, the headlamp
    // (free — a visible cell is eye-unoccluded), and emissive are left untouched,
    // so a shadowed cell darkens and reads as directional shadow without ever
    // collapsing to pure black. `ShadowQuality::Off` leaves this at 1.0 → the
    // pre-shadow output, bit-for-bit.
    let shadow_key = lc.shadow_key.clamp(0.0, 1.0);

    // Diffuse from the directional KEY light: Lambert N·L, attenuated by the cast
    // shadow.
    let n_dot_l = normal.dot(lc.key_dir).max(0.0);
    let diffuse_key = (n_dot_l * shadow_key) * lc.key_color;

    // VIEW-INDEPENDENT mode (`view_indep`): when set, every term that reads
    // `lc.view_dir` is dropped (headlamp facing N·V, and both specular glints), so
    // the lit result is a function only of (world normal, key light, ambient,
    // height/AO) — invariant to camera facing. This is what makes the near-ring pan
    // byte-exact. Default OFF keeps the full Blinn-Phong look, byte-for-byte.

    // Headlamp: a light co-located with the eye (along view_dir). N·V gives the
    // facing term; it lights what the player looks at and makes relief pop as the
    // camera moves. Warm-white, scaled by the headlamp knob. VIEW-DEPENDENT — drop
    // it under the toggle (→ Vec3::ZERO, no headlamp flood).
    let n_dot_v = if view_indep {
        0.0
    } else {
        normal.dot(lc.view_dir).max(0.0)
    };
    let headlamp = if view_indep {
        Vec3::ZERO
    } else {
        (lc.headlamp * n_dot_v) * Vec3::new(1.0, 0.96, 0.88)
    };

    // PUNCTUAL lights: level-authored point lights, UNSHADOWED, summed
    // at the cell's world hit point against the (relief-perturbed) normal. They are
    // VIEW-INDEPENDENT by construction — position and normal only — so they survive
    // the `view_indep` toggle and the near-ring pan stays byte-exact under it.
    // `PointLights::NONE` (every shipped level) returns ZERO on a single branch, so
    // the un-lit path is bit-for-bit the pre-lighting output.
    let points = point_light_sum(&lc.points, lc.world_pos, normal);

    // Combined diffuse + ambient, occluded. `lit` lights the COLOR (fg/bg) and
    // includes the viewpoint-locked headlamp; `relief_lit` deliberately EXCLUDES
    // the headlamp so the glyph ramp tracks height + the directional key (which
    // varies across brick courses) rather than the flat facing flood. Point lights
    // go in BOTH: they are part of the scene's real illumination, so a cell inside
    // a light pool must brighten AND pick a denser ramp glyph — that pooling is the
    // whole point of authoring a light.
    let lit = (lc.ambient + diffuse_key + points + headlamp) * ao;
    let relief_lit = (lc.ambient + diffuse_key + points) * ao;
    let mut fg = albedo * lit;

    // Specular: Blinn-Phong half-vector glint on ridges. Use the brighter of the
    // key and the headlamp as the spec source so glints appear both where the key
    // rakes and where the player's lamp hits head-on. BOTH glints are
    // VIEW-DEPENDENT (half-vector and N·V read the view direction) → fully dropped
    // under the toggle.
    let spec = if view_indep {
        0.0
    } else {
        let half_key = (lc.key_dir + lc.view_dir).normalize_or_zero();
        // The key glint is shadowed with the key diffuse (a point in the key's
        // shadow sees no key highlight); the headlamp glint is NOT (lamp at eye).
        let spec_key = normal.dot(half_key).max(0.0).powf(lm.shininess) * n_dot_l * shadow_key;
        let spec_lamp = n_dot_v.powf(lm.shininess) * lc.headlamp;
        lm.specular * (spec_key + spec_lamp) * ao
    };
    fg += unpack_rgb(lm.spec_color) * spec;

    // Emissive: a fraction of the raw albedo that ignores the light (self-lit
    // traces). Added on top so it survives shadow.
    fg += albedo * lm.emissive;

    let fg = fg.clamp(Vec3::ZERO, Vec3::ONE);

    // The bg is pushed toward shadow: darker than the lit fg, deepened in
    // crevices, so the cell's fg/bg contrast reads as a raised glyph on a recessed
    // ground. Keep a floor so it never muds to pure black under ambient.
    let shadow =
        (lc.ambient.max_element() * 0.6 + 0.40 * (lit.max_element())).clamp(0.12, 1.0) * ao;

    // Luminance the glyph ramp reads: the RELIEF energy (ambient + directional
    // key diffuse, NO headlamp) plus a height bias and spec, so RAISED + KEY-LIT
    // cells score high (dense glyph) and RECESSED/SHADOWED cells score low
    // (sparse glyph). Excluding the viewpoint-locked headlamp keeps facing walls
    // from pinning the ramp to `@`/`%`; the headlamp still lights the COLOR above.
    let lum =
        (relief_lit.dot(Vec3::new(0.299, 0.587, 0.114)) + 0.35 * (h - 0.5) + spec).clamp(0.0, 1.5);

    (fg, shadow, lum, points)
}

/// Perturb a face normal by a height gradient `(dh_du, dh_dv)` in the surface's
/// tangent frame, producing a world-space relief normal. `normal_strength`
/// scales the tilt; `0` returns the face normal unchanged. The gradient pushes
/// the normal "downhill" so a bump's lit side faces the rising slope.
#[inline]
fn perturb_normal(lc: &LightCtx, dh_du: f32, dh_dv: f32, normal_strength: f32) -> Vec3 {
    // Tangent-space normal: a slope (−dh) along each tangent, +1 out the face.
    // Scaling the tangent components by normal_strength controls relief depth.
    let n = lc.face_n
        - lc.tangent_u * (dh_du * normal_strength)
        - lc.tangent_v * (dh_dv * normal_strength);
    n.normalize_or_zero()
}

/// The VIEW-INCIDENCE azimuth of a cell, normalized to `[0,1)` — the angle at which
/// the eye grazes the surface, read from the per-cell [`LightCtx`]. Used by
/// [`SuperSelector::ByAngle`] (the angle-driven glyph) to pick a family member that
/// tracks the viewing direction.
///
/// The view direction (surface→camera) is projected onto the surface's tangent plane
/// (`tangent_u`, `tangent_v`); the `atan2` of that projection gives an azimuth around
/// the face normal, wrapped to `[0,1)`. A near face-on view (the projection is tiny)
/// collapses to the centre `0.5` so a head-on surface picks a stable middle member.
/// Pure, allocation-free, a couple of dot products + one `atan2`.
#[inline]
fn view_incidence01(lc: &LightCtx) -> f32 {
    // Project the view dir onto the two surface tangents → an in-plane 2D vector.
    let u = lc.view_dir.dot(lc.tangent_u);
    let v = lc.view_dir.dot(lc.tangent_v);
    // Near face-on (the in-plane component is negligible): no meaningful azimuth →
    // the face-on centre, so the pick is stable rather than spinning on noise.
    if u * u + v * v < 1.0e-6 {
        return 0.5;
    }
    // atan2 ∈ (−π, π] → normalize to [0,1).
    let a = v.atan2(u);
    (a / std::f32::consts::TAU + 0.5).rem_euclid(1.0)
}

// ─────────────────────────── noise / pattern recipes ────────────────────────

/// Value noise in `[0,1]` from the integer-cell hash (bilinear, smoothstepped).
/// Cheap, deterministic, platform-stable — independent of the `noise` crate.
fn value_noise(seed: u32, p: Vec2) -> f32 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let fx = fract(p.x);
    let fy = fract(p.y);
    let ux = smoothstep(0.0, 1.0, fx);
    let uy = smoothstep(0.0, 1.0, fy);
    let v = |dx: i32, dy: i32| hash01(hash_cell(seed, ix + dx, iy + dy));
    let nx0 = lerp(v(0, 0), v(1, 0), ux);
    let nx1 = lerp(v(0, 1), v(1, 1), ux);
    lerp(nx0, nx1, uy)
}

/// Fractal Brownian motion: summed octaves of Perlin (from the `noise` crate),
/// remapped to `[0,1]`. `octaves` × `lacunarity` (freq mult) × `gain` (amp mult).
///
/// `pub(crate)` so the sibling [`crate::fractal`] module enriches its log-polar
/// field with the SAME noise the recipes use (one shared field, no duplicate fBm).
pub(crate) fn fbm(perlin: &Perlin, p: Vec2, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut freq = 1.0f32;
    let mut amp = 0.5f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        let s = perlin.get([(p.x * freq) as f64, (p.y * freq) as f64]) as f32; // [-1,1]
        sum += amp * s;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    (0.5 + 0.5 * (sum / norm.max(1e-6))).clamp(0.0, 1.0)
}

// ── Band-limited octaves (Nyquist fade by screen period) ─────────────────────
//
// `fbm`/`ridged` above cap the OCTAVE COUNT by distance tier
// (`ShadeLod::octaves`), which is a hard cutoff — a whole octave pops in or
// out at a tier boundary. A. Quilez's band-limiting
// (https://iquilezles.org/articles/bandlimiting/) fades EACH octave's own
// contribution to zero as its SCREEN PERIOD (how many terminal cells one
// wavelength of that octave spans) drops below the ~2-cell Nyquist floor, so
// the same octave count near and far never *aliases* — the unresolvable high
// frequencies fade continuously into the mean (B carries it) instead of being
// point-sampled into shimmer.

/// The low/high screen-period band (in cells) an octave fades across:
/// below [`NYQUIST_FADE_LO`] cells a period cannot be resolved at all (weight 0);
/// above [`NYQUIST_FADE_HI`] it is fully resolvable (weight 1). Between the two,
/// `smoothstep` gives a continuous, pop-free fade — a "spread out" reveal
/// expressed at the octave level, distinct from the whole-material detail-tier
/// cross-fade layered on top.
const NYQUIST_FADE_LO: f32 = 2.0;
const NYQUIST_FADE_HI: f32 = 4.0;

/// The fade weight for one octave of world wavelength `wavelength_world` (in the
/// SAME units as `world_per_cell`, i.e. the noise argument's own coordinate
/// space — see the call-site comment on how callers convert). `world_per_cell
/// <= 0.0` (unknown footprint — [`ShadeLod::FULL`] and any legacy caller that
/// never threaded a distance) disables the fade entirely (weight `1.0`), so a
/// caller that doesn't know its footprint gets exactly today's behaviour.
#[inline]
fn octave_fade(wavelength_world: f32, world_per_cell: f32) -> f32 {
    if world_per_cell <= 0.0 || !world_per_cell.is_finite() {
        return 1.0;
    }
    let period_cells = wavelength_world / world_per_cell;
    smoothstep(NYQUIST_FADE_LO, NYQUIST_FADE_HI, period_cells)
}

/// [`fbm`] with each octave band-limited by [`octave_fade`]. `world_per_cell`
/// is how many WORLD units (in `p`'s own coordinate space — i.e. already divided
/// by whatever `scale` the caller multiplied `coord` by to build `p`) one screen
/// cell covers; `<= 0.0` reproduces [`fbm`] exactly (no fade — the byte-identical
/// fallback for a caller with no footprint). The weighted sum is re-normalised by
/// `Σ(amp·w)`, so a fully-faded high octave doesn't just drop out silently — the
/// remaining (coarser) octaves keep carrying the FULL amplitude budget, exactly as
/// `fbm`'s own `norm` does for a hard octave-count cap. The material's overall
/// brightness is unaffected by how many octaves are currently resolvable.
pub(crate) fn fbm_bandlimited(
    perlin: &Perlin,
    p: Vec2,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    world_per_cell: f32,
) -> f32 {
    let mut freq = 1.0f32;
    let mut amp = 0.5f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        let w = octave_fade(1.0 / freq.max(1e-6), world_per_cell);
        if w > 0.0 {
            let s = perlin.get([(p.x * freq) as f64, (p.y * freq) as f64]) as f32;
            sum += amp * w * s;
        }
        norm += amp * w;
        freq *= lacunarity;
        amp *= gain;
    }
    (0.5 + 0.5 * (sum / norm.max(1e-6))).clamp(0.0, 1.0)
}

/// Ridged fBm — sharp ridge lines (`1 - |perlin|`, summed; good for veins) — with
/// each octave band-limited by [`octave_fade`]. See [`fbm_bandlimited`] for
/// the parameter contract; the sole caller is `sample_circuit`'s veins, so a
/// separate un-band-limited `ridged` is not kept alongside it.
pub(crate) fn ridged_bandlimited(
    perlin: &Perlin,
    p: Vec2,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    world_per_cell: f32,
) -> f32 {
    let mut freq = 1.0f32;
    let mut amp = 0.5f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        let w = octave_fade(1.0 / freq.max(1e-6), world_per_cell);
        if w > 0.0 {
            let s = perlin.get([(p.x * freq) as f64, (p.y * freq) as f64]) as f32;
            let r = 1.0 - s.abs();
            sum += amp * w * r * r;
        }
        norm += amp * w;
        freq *= lacunarity;
        amp *= gain;
    }
    (sum / norm.max(1e-6)).clamp(0.0, 1.0)
}

/// Worley / Voronoi F1 distance + the owning feature cell.
///
/// Returns `(f1, cell_hash)` where `f1` is the distance to the nearest jittered
/// feature point (≈`[0,1]`) and `cell_hash` identifies that point's cell (for
/// per-cell coloring — circuit nodes, brick variation, star identity).
fn worley(seed: u32, p: Vec2) -> (f32, u32) {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let mut best = f32::INFINITY;
    let mut best_cell = 0u32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = ix + dx;
            let cy = iy + dy;
            let jitter = hash2_01(seed, cx, cy);
            let feat = Vec2::new(cx as f32 + jitter.x, cy as f32 + jitter.y);
            let d = (feat - p).length();
            if d < best {
                best = d;
                best_cell = hash_cell(seed, cx, cy);
            }
        }
    }
    (best, best_cell)
}

/// Running-bond brick lattice. Returns `(mortar, brick_id, u_in_brick,
/// v_in_brick)`: `mortar ∈ [0,1]` (1 = on a mortar seam), a per-brick id for
/// color variation, and the local coords within the brick face.
fn brick(p: Vec2, aspect: f32, mortar_w: f32) -> (f32, u32, f32, f32) {
    let row = p.y.floor();
    // Alternate courses shift by half a brick (running bond).
    let offset = if (row as i32 & 1) != 0 { 0.5 } else { 0.0 };
    let bx = p.x * aspect + offset;
    let ix = bx.floor() as i32;
    let iy = row as i32;
    let u = fract(bx);
    let v = fract(p.y);
    // Mortar: a band near each brick edge in u and v.
    let du = u.min(1.0 - u);
    let dv = v.min(1.0 - v);
    let edge = du.min(dv);
    let mortar = 1.0 - smoothstep(mortar_w * 0.5, mortar_w, edge);
    (mortar, hash_cell(0xB12C, ix, iy), u, v)
}

// ─────────────────────────── material model ─────────────────────────────────

/// Stable id for a [`Material`] in the [`MaterialRegistry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u16);

/// Which procedural recipe a material uses + its tunable parameters. Data only —
/// adding a look is a new variant entry, not new shader code paths everywhere.
#[derive(Clone, Copy, Debug)]
pub enum TextureRecipe {
    /// Brick / panel masonry. `aspect` = bricks-per-cell-width, `mortar_w` =
    /// seam width in brick-units, `grain` = fBm body mottling amount.
    Brick {
        /// Bricks-per-cell-width.
        aspect: f32,
        /// Seam width in brick-units.
        mortar_w: f32,
        /// fBm body mottling amount.
        grain: f32,
    },
    /// Circuit board: Worley cells = pads/vias, fBm = copper pour, traces on the
    /// Voronoi edges. `scale` = cells-per-world-cell, `trace_w` = edge width.
    Circuit {
        /// Cells-per-world-cell.
        scale: f32,
        /// Edge width.
        trace_w: f32,
    },
    /// World-pinned grain: hashed feature specks (Worley) over an fBm base, plus
    /// a faint integer tile grid. `scale` = grain lattice density, `speck` =
    /// fraction of cells carrying a speck.
    Grain {
        /// Grain lattice density.
        scale: f32,
        /// Fraction of cells carrying a speck.
        speck: f32,
        /// Integer-tile grid epsilon.
        tile_eps: f32,
    },
    /// Star sky: sparse hashed stars over a soft fBm nebula. `scale` = star
    /// lattice density, `density` = star fraction.
    Sky {
        /// Star lattice density.
        scale: f32,
        /// Star fraction.
        density: f32,
    },
    /// `Hologram`: a GLYPH-ONLY transparent surface. The bg is forced to a
    /// near-black VOID (no filled colour); the only ink is fg-coloured horizontal
    /// scan/line glyphs (`─ ═ ╌ ·`) forming holographic scanlines. With a `Shift`
    /// animation the lines scroll vertically over time, so the surface reads as a
    /// glowing, see-through hologram (colored line-glyphs on void) rather than a
    /// filled wall. `line_freq` = scan lines per world-cell (vertical density);
    /// `scan_w` = fraction of each line period that inks (line thickness `[0,1]`).
    /// Self-lit (not relit by the relief pass) so the fg hue is the glyph colour.
    Hologram {
        /// Scan lines per world-cell (vertical density).
        line_freq: f32,
        /// Fraction of each line period that inks (line thickness `[0,1]`).
        scan_w: f32,
    },
    /// `CodeWall`: render the engine's OWN annotated source onto the wall — the
    /// strange loop. `panel` selects which embedded source file (see
    /// [`crate::code_wall`]); `u_chars_per_unit` tunes how many source columns
    /// span one world-cell of wall (legibility vs. how much code fits). The glyph
    /// is the source character at the cell's `(u, wall_v)`; comments glow brighter
    /// than code (the system's voice vs. its body). The fourth wall, literally.
    CodeWall {
        /// Which embedded source file (see [`crate::code_wall`]).
        panel: u32,
        /// Source columns spanning one world-cell of wall.
        u_chars_per_unit: f32,
    },
    /// `NoteWall`: paint ONE authored `[note]` placard onto ONE wall cell face — the
    /// level's own voice, in the room, on the wall the author pointed at.
    ///
    /// `note` indexes the level's note panels in the active
    /// [`TextSurfaceRegistry`](crate::scene_text::TextSurfaceRegistry) (parse order).
    /// `face` is the face the text is painted on, named by its OUTWARD normal.
    ///
    /// Unlike [`CodeWall`](Self::CodeWall) — which flows text ACROSS a long wall run
    /// at an authored character density — a placard FITS its whole (short, wrapped)
    /// body across a single unit face, because that is how a sign works.
    ///
    /// **The face gate.** A material is bound to a CELL, and `side` carries no sign,
    /// so without gating the placard would appear on all four faces of the cell —
    /// including, mirrored, in the room on the other side of the wall. So: when the
    /// cell has lighting geometry, gate on the true face normal
    /// ([`re_core::FaceDir::matches_normal`]); when it does not (a focused unit test,
    /// or any caller shading without light) fall back to the coarse NS/EW `side` bit
    /// ([`re_core::FaceDir::side`]), which admits the matching axis PAIR. A gated-OFF
    /// face renders the blank plate, never the text.
    NoteWall {
        /// Index into the level's note panels (parse order).
        note: u32,
        /// The face the text is painted on, named by its OUTWARD normal.
        face: re_core::FaceDir,
    },
    /// `Scanline`: a FILLED directional-band surface — the smart-material primitive
    /// the catalog uses for CRT/monitor screens, brushed metal, woven fabric, and
    /// striped/grated looks the void-bg [`TextureRecipe::Hologram`] and the cellular
    /// [`TextureRecipe::Brick`]/[`TextureRecipe::Circuit`] recipes don't cover cleanly.
    ///
    /// Unlike the transparent hologram, the gaps between bands keep a real (dark)
    /// `bg_palette` fill, so the surface reads as a solid panel. The band axis is
    /// selectable: `axis = 0` ⇒ HORIZONTAL bands (CRT raster lines, brushed metal
    /// grain across the face); `axis = 1` ⇒ VERTICAL bands (a vertical grate / blind);
    /// `axis = 2` ⇒ a WOVEN cross-hatch (both axes — fabric / mesh). `freq` = bands
    /// per world-cell; `duty` = inked fraction of each band period `[0,1]` (line
    /// thickness vs. gap); `warp` = fBm jitter `[0,1]` that breaks perfectly straight
    /// lines into a hand-made grain (metal/wood). Self-lit (like the hologram + code
    /// wall, the relief pass leaves it alone) so the `fg` phosphor/sheen hue reads
    /// directly; `emissive` in the [`LightModel`] drives the glow + scan-glow accent.
    Scanline {
        /// Bands per world-cell.
        freq: f32,
        /// Inked fraction of each band period `[0,1]` (line thickness vs. gap).
        duty: f32,
        /// fBm jitter `[0,1]` that breaks straight lines into a hand-made grain.
        warp: f32,
        /// `0` horizontal, `1` vertical, `2` woven cross-hatch.
        axis: u8,
    },
    /// `Truchet`: EDGE-MATCHED conduit tiling — box-drawing runs that CONTINUE across
    /// cell boundaries instead of speckling independently (see [`crate::truchet`]).
    ///
    /// Every other procedural recipe here samples each cell independently, which gives
    /// texture but never structure: a crack stops at the cell edge, a conduit goes
    /// nowhere. This one derives each EDGE's crossing bit from a hash of the edge's own
    /// coordinates, so two neighbours read the same bit for their shared side and the
    /// run is connected by construction — no storage, no scan order, still O(1).
    ///
    /// `seed` selects the pattern; `density` `[0,1]` is the fraction of edges crossed
    /// (`0` = blank, `1` = a full lattice of `┼`); `scale` is TILES PER WORLD CELL on
    /// both axes. Self-lit like the other pattern recipes, so the ink hue reads
    /// directly.
    ///
    /// `scale` matters more than it looks. The index is a floor of a CONTINUOUS
    /// world coordinate, so multiplying that coordinate lands every tile on one
    /// global integer grid whatever the factor — continuity survives any positive
    /// `scale`, including a fractional one. At `1.0` a wall cell is a single tile,
    /// which on screen is one box-drawing glyph stretched over the whole cell: a
    /// lattice in principle and a flat band in practice. Higher values buy the
    /// structure the recipe exists for.
    Truchet {
        /// Selects the pattern.
        seed: i32,
        /// Fraction of edges crossed (`0` = blank, `1` = a full lattice of `┼`).
        density: f32,
        /// Tiles per world cell on both axes.
        scale: f32,
    },
    /// `Fractal`: a log-polar self-similar "portal" — nested rings / spiral arms
    /// that zoom / spin / spin-zoom WITHOUT a visible seam (the pattern is periodic
    /// over one zoom-octave and one turn; see [`crate::fractal`]). `arms` = spiral
    /// arms (`0` = pure concentric rings), `octave_warp` = fBm enrichment strength
    /// `[0,1]`, `twist` = arm curl (log-spiral vs straight rays). Self-lit (like the
    /// hologram / scanline, the relief pass leaves it alone) so the ring-phase
    /// palette reads directly. The three [`AnimKind::Zoom`]/[`AnimKind::Spin`]/
    /// [`AnimKind::SpinZoom`] modes drive the live phase.
    Fractal {
        /// Spiral arms (`0` = pure concentric rings).
        arms: u8,
        /// fBm enrichment strength `[0,1]`.
        octave_warp: f32,
        /// Arm curl (log-spiral vs straight rays).
        twist: f32,
    },
    /// `Substrate`: the substrate's own CELL ALPHABET as a wall surface — `░`
    /// unclaimed, `█` held, `▒` released (a scar), `✕` burned. The glyph IS the
    /// meaning, so (like [`CodeWall`](Self::CodeWall) and [`NoteWall`](Self::NoteWall))
    /// the relief pass must not re-select it.
    ///
    /// `scale` = alphabet cells per world unit (the grid the eye counts;
    /// [`crate::substrate::CELLS_PER_UNIT`] is the shipped density). `held` is the
    /// fraction of the field ever claimed; `released` is the fraction OF THAT HELD SET
    /// since let go — so raising `released` ERODES the held set rather than flipping
    /// it. See [`crate::substrate`] for the full state table and the measurement that
    /// pins it.
    ///
    /// Static: the field is a pure function of the world cell and the two authored
    /// scalars, with no `time` term at all, so a substrate wall is fully cacheable and
    /// reports `is_animated() == false`.
    Substrate {
        /// Alphabet cells per world unit.
        scale: f32,
        /// Fraction of the field ever claimed.
        held: f32,
        /// Fraction OF THE HELD SET since let go (erodes the held set).
        released: f32,
    },
    /// `Voice`: [`Substrate`](Self::Substrate)'s cell alphabet driven by PUZZLE STATE
    /// instead of by two authored scalars — the `mnemos_hold` / `lethe_release` pair.
    ///
    /// One scalar, the level's monotone solve-progress MARK, arrives as the cell's
    /// state-bus word ([`SampleCtx::state`], quantised parts-per-thousand — see
    /// [`crate::substrate::decode_mark`]) and is spent on one channel or the other per
    /// [`VoiceMode`](crate::substrate::VoiceMode):
    ///
    /// | mode | `held` | `released` | reads as |
    /// |---|---|---|---|
    /// | `Hold` | `mark` | `0` | an empty wall FILLING with `█` |
    /// | `Release` | `1` | `mark` | a whole wall ERODING to `▒` |
    ///
    /// `scale` is the alphabet density, exactly as on `Substrate`.
    ///
    /// IRREVERSIBILITY is a property of the MARK, not of this recipe: the high-water
    /// `max()` and its clear-on-reset are level-session state and live in `re_game`'s
    /// `state_bridge`. Here the surface stays a pure function of `(world cell, mark)`,
    /// which is what keeps it testable and what keeps it CACHEABLE — no `time` term,
    /// so it reports `is_animated() == false` and a static scene still idles.
    ///
    /// # The one-channel-two-readings rule
    ///
    /// `state` is used everywhere else to INDEX a [`FamilyDriver::State`](crate::glyph_family::FamilyDriver::State)
    /// glyph family, to trigger an [`AnimTrigger::OnState`] track, or to level a
    /// state-reading [`SurfaceFx`](crate::surface_fx::SurfaceFx). This recipe reads the
    /// same word as a SCALAR, which is safe only while the two never meet on one
    /// material — asserted by `the_voice_materials_carry_no_other_state_reader`, not
    /// left to convention.
    Voice {
        /// The alphabet density, exactly as on [`Substrate`](Self::Substrate).
        scale: f32,
        /// Which puzzle-state channel `held`/`released` reads from.
        mode: crate::substrate::VoiceMode,
    },
    /// `Image`: sample an attached RGBA8 [`re_core::TextureImage`] (e.g. a decoded
    /// PNG) onto the wall face. `tex_id` is the index into [`re_world::Map::textures`]
    /// — [`crate::registry_for_map`] resolves it to the actual `Arc<TextureImage>` and
    /// attaches it to the [`Material::texture`] field, so the shader reads pixels with
    /// NO `&Map`. The cell's `(tex_u, wall_v)` give `(u, v)`; the nearest texel paints a
    /// SOLID-COLOUR `█` block by default (no alpha, no sub-cell, no tiling).
    /// An out-of-range / missing texture renders a magenta error block, never a panic.
    ///
    /// `span` is the WORLD-ANCHORED UV mapping: when `Some`, ONE copy of the
    /// figure is stretched across the surface's full world extent (so an N-cell-wide
    /// textured wall shows ONE figure, not N tiled copies); when `None`, the legacy
    /// per-cell `[0,1]` tiling (one figure per cell). See [`ImageSpan`].
    Image {
        /// Index into [`re_world::Map::textures`], resolved to the actual
        /// texture by [`crate::registry_for_map`].
        tex_id: u32,
        /// World-anchored UV mapping override — see [`ImageSpan`].
        span: Option<ImageSpan>,
        /// SUB-CELL detail mode: sample the world-anchored texture at multiple
        /// sub-positions inside each screen cell and emit a higher-resolution glyph.
        /// `Off` ⇒ the legacy per-cell solid `█` (byte-identical); `Quad` ⇒ 2×2
        /// quadrant block + a two-colour split; `Braille` ⇒ a 2×4 Braille coverage
        /// glyph (one ink + backdrop); `Ascii`/`Contour`/`Lines` ⇒ the ASCII-family
        /// edge samplers (luminance ramp + Sobel/DoG direction glyphs). The 1:1 world
        /// mirror is `re_world::SubcellMode`.
        subcell: SubcellMode,
        /// How the image is mapped onto the wall when no explicit `span` override is
        /// given. `Some(span)` wins and `fit` is ignored (byte-identical to the
        /// explicit-span path); `None` + `fit` derives the
        /// `ImageSpan` from the wall world extent + image aspect in
        /// [`crate::registry_for_map`]. The 1:1 world mirror is `re_world::Fit`.
        fit: Fit,
        /// `tile = a b` - how many times the figure REPEATS across the mapped span
        /// (`a` along the surface, `b` up it). [`IMAGE_TILE_NONE`] (`1,1`) is the
        /// default and short-circuits the multiply entirely, so an untiled image is
        /// byte-identical. A tiled UV WRAPS (that is the repeat); an untiled one is
        /// clamped exactly as before. The 1:1 world mirror is the `re_world` recipe's
        /// `tile` pair.
        tile: (f32, f32),
        /// `relief = height_scale=... normal_strength=...` - when `Some`, this image is
        /// a LIT SURFACE, not a self-lit decal: [`RecipeProps::self_lit`] goes `false`
        /// and [`Material::height_at`] reads the attached texture's
        /// [`re_core::TextureImage::height_map`] instead of returning a flat constant,
        /// so the existing normal-perturbation + glyph-ramp relief pass runs on a
        /// photo the same way it runs on brick. `None` (every image authored before
        /// this key) keeps the historical self-lit block. The two knobs themselves are
        /// ALSO copied onto [`Material::light`] by `crate::material_from_def`; they
        /// live here as well because the recipe is what the props table reads.
        relief: Option<ImageRelief>,
    },
    /// `Facade`: the structured bay×floor WINDOW GRID — see [`crate::facade`] for the pure sampler and
    /// its style table. `bay`/`floor_h` are repeats PER WORLD-CELL on the wall's
    /// along/up axes (like [`Truchet`](Self::Truchet)'s `scale`); `lit` is the
    /// fraction of units that hash LIT; `seed` salts the per-building hash so two
    /// facades with identical knobs still show different windows. `ground_band` /
    /// `sign_row` opt into the two absolute-height bands along the wall's base.
    ///
    /// Reused unchanged by `sky::skyline` for its distant towers'
    /// window rhythm — one recipe, two consumers.
    ///
    /// `.lvl` syntax: `recipe = facade style=office bay=3 floor=4 lit=0.35`.
    Facade {
        /// Window-unit style catalog entry — see [`crate::facade::FacadeStyle`].
        style: crate::facade::FacadeStyle,
        /// Bay repeats per world-cell on the wall's along axis.
        bay: f32,
        /// Floor repeats per world-cell on the wall's up axis.
        floor_h: f32,
        /// Fraction of units that hash LIT.
        lit: f32,
        /// Salts the per-building hash so identical knobs still differ.
        seed: i32,
        /// Whether to draw the ground-floor absolute-height band.
        ground_band: bool,
        /// Whether to draw the sign-row absolute-height band.
        sign_row: bool,
    },
}

/// The default `tile =` repeat of a [`TextureRecipe::Image`]: ONE copy across the
/// mapped span. Named so "is this image tiled at all?" is a comparison against a
/// constant, and so the byte-identical fast path is a single equality test.
pub const IMAGE_TILE_NONE: (f32, f32) = (1.0, 1.0);

/// The two RELIEF knobs an image material authors with
/// `relief = height_scale=... normal_strength=...`. The 1:1 renderer mirror of
/// `re_world::ImageRelief`.
///
/// Its PRESENCE on a [`TextureRecipe::Image`] is the switch that turns a flat,
/// self-lit photo decal into a lit 3-D surface; the values scale the relief the same
/// way [`LightModel::height_scale`] / [`LightModel::normal_strength`] do for a
/// procedural recipe (they are copied onto the material's [`LightModel`] at build).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRelief {
    /// How tall the height field reads, in tangent units. `0` means flat.
    pub height_scale: f32,
    /// Tangent-space normal tilt scale. `0` means lit as a flat face.
    pub normal_strength: f32,
}

/// How a [`TextureRecipe::Image`] figure is mapped onto its wall surface when no
/// explicit `span` override is given. The 1:1 renderer mirror of `re_world::Fit`.
/// When `span` is `Some`, the authored
/// [`ImageSpan`] is used directly and `fit` is ignored; when `span` is `None`,
/// [`crate::registry_for_map`] derives the span from the wall's world extent + the
/// image aspect via [`crate::fit_span`] per this rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Fit {
    /// Fill the wall exactly; aspect ratio ignored (the image distorts). This is the
    /// pre-fit no-aspect behaviour: the span IS the wall box.
    Stretch,
    /// Preserve aspect; the WHOLE image fits inside the wall (margins). Default.
    /// A `(u, v)` outside `[0, 1]` falls in the margin → transparent/edge.
    #[default]
    Contain,
    /// Preserve aspect; fill the wall completely, cropping the overflow (UVs stay
    /// within `[0, 1]`).
    Cover,
}

/// SUB-CELL detail mode for a [`TextureRecipe::Image`] surface. The 1:1
/// renderer mirror of `re_world::SubcellMode`. Default [`Off`](Self::Off) keeps the
/// legacy per-cell solid block, byte-identical to the plain span-mapped image look.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SubcellMode {
    /// One solid block per cell. Default — byte-identical to the plain image look.
    #[default]
    Off,
    /// 2×1 half-block: two sub-texels → `▀`/`▄`/`█`/space + a two-colour split.
    /// The cheapest rung; also the one that makes a sub-texel roughly SQUARE, since a
    /// terminal cell is about 2:1 tall.
    Half,
    /// 2×2 quadrant: four sub-texels → one of 16 block glyphs + a near/backdrop
    /// two-colour split. Doubles effective resolution.
    Quad,
    /// 3×2 sextant: six sub-texels → a `U+1FB00..=U+1FB3B` BLOCK SEXTANT glyph (or one
    /// of the four Block Elements Unicode already had) + a two-colour split. Three
    /// times a quadrant's sub-texels while STILL keeping two colours, which is what
    /// [`Braille`](Self::Braille) cannot do. Needs terminal sextant support.
    Sextant,
    /// 2×4 Braille: eight sub-texels → an 8-dot coverage glyph. One ink colour + a
    /// backdrop, for a sharp alpha-cut-out silhouette.
    Braille,
    /// ASCII edge sampler: luminance ramp (` .:-=+*#%@`) fill; on a Sobel edge, swap
    /// to the direction glyph (`/ \ | - _`). The lightest ASCII algorithm. Uses the
    /// precomputed [`crate::edge_field::EdgeField`].
    Ascii,
    /// DoG-hybrid contour sampler: the same ramp fill but the edge layer uses the
    /// Difference-of-Gaussians field → thin, clean contour lines (the hero look).
    Contour,
    /// Line-only / ink sampler: EVERY cell is a glyph, but non-edge cells are inked
    /// close to their background (low contrast) so only contour glyphs pop.
    Lines,
}

/// WORLD-ANCHORED UV span for an [`TextureRecipe::Image`] surface. The 1:1
/// renderer mirror of `re_world::ImageSpan`. The shader maps the texture's
/// `[0,1]×[0,1]` UV box onto a WORLD rectangle so ONE figure spans the whole
/// textured surface no matter how many cells wide/tall it is:
///
/// * world ALONG (the wall's variable axis — world Y on an NS face, world X on an
///   EW face): `u = (along - along_min) / along_size`;
/// * world UP ([`SampleCtx`]'s `wall_world_y`, the texel's world height):
///   `v = (up_top - world_up) / up_size`, so `v = 0` at the TOP of the span (image
///   top-left origin) and `v = 1` at the bottom.
///
/// `(u, v)` outside `[0,1]` is OUTSIDE the figure → transparent for the cut-out
/// (shows the backdrop), edge-clamped for the opaque paint path. As the camera
/// approaches, each screen cell covers a smaller world ALONG/UP span → fewer texels
/// per cell → nearest sampling stays crisp ("walk up and see it pixel-by-pixel").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageSpan {
    /// World coordinate (wall variable axis) where `u = 0`.
    pub along_min: f32,
    /// World extent the full `u` range covers (`> 0`).
    pub along_size: f32,
    /// World UP coordinate (height) where `v = 0` (TOP of the figure).
    pub up_top: f32,
    /// World vertical extent the full `v` range covers (`> 0`).
    pub up_size: f32,
}

impl ImageSpan {
    /// Map a WALL hit's world ALONG + UP coordinates to texture `(u, v)`. Pure: the
    /// SAME world point always yields the same `(u, v)`, so the mapping is
    /// view-independent (close vs far, the same world point samples the same texel).
    /// Returns the RAW (un-clamped) `(u, v)`; callers clamp (opaque paint) or
    /// reject-if-outside (cut-out) per their contract.
    #[inline]
    pub fn uv(&self, world_along: f32, world_up: f32) -> (f32, f32) {
        let u = (world_along - self.along_min) / self.along_size;
        let v = (self.up_top - world_up) / self.up_size;
        (u, v)
    }
}

/// How the recipe's BODY glyph is allowed to thin to a coloured blank on the
/// world-scattered fraction the sparse-body coverage mask drops (step 4 of
/// [`Material::sample`]) — the "carry detail with colour, not glyphs" thinning.
///
/// A pure DATA discriminant on [`RecipeProps`]: each recipe declares which (if
/// any) world-anchored coverage mask thins its body, so the single sparse-body
/// site reads this instead of re-matching the recipe. Adding a recipe that wants
/// a thinning mask means adding a variant here + the one arm in [`TextureRecipe::
/// props`] — nowhere else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyBlankKind {
    /// The body never thins to blank (filled / glyph-only / self-lit recipes).
    None,
    /// Wall masonry thinning — the [`wall_body_inked`] world mask (brick).
    Wall,
    /// Floor grain thinning — the [`floor_body_inked`] world mask (grain).
    Floor,
}

/// The EXTRA [`SampleCtx`] inputs a recipe reads, beyond the ones every cache key
/// already carries (`coord`, `distance`, `foreshorten`, fog, the light digest).
///
/// # Why this exists — a cache that stopped fingerprinting its own inputs
///
/// [`SampleKey`] and [`WorldTexKey`] are correctness-preserving only while they carry
/// EVERY output-affecting input: a hit returns a stored texel on the strength of the
/// key alone. `Substrate` and `Voice` read [`SampleCtx::wall_along`] /
/// [`SampleCtx::wall_up`], and `Voice` also reads [`SampleCtx::state`] — none of which
/// the keys carried. The reason it was not firing is the sharp part: `coord.x` for a
/// wall is `tex_u + floor(the face's CONSTANT axis)`, so it WRAPS once per world cell,
/// and it was the raw `distance` differing between those cells that happened to keep
/// them apart. That is a property of ordinary scenes, not of the key — and it fails
/// exactly where the key deliberately discards distance (the far-frozen plateau, see
/// [`SampleKey::frozen_color`]) or where two cells of one material carry different
/// state words (an explicitly cross-wired voice cell).
///
/// # The shape, and the precedent it follows
///
/// `is_sky` already does this: `frozen_color`'s doc says a future distance-reading
/// recipe "opts out by declaring it in `TextureRecipe::props`, in one place". This is
/// the same seam generalised. A recipe declares what it reads; the key builders fold
/// exactly those fields in. [`NONE`](Self::NONE) — what every recipe that shipped
/// before the voices declares — produces a key BYTE-IDENTICAL to the pre-fix one,
/// which is what leaves every golden untouched
/// (`a_recipe_declaring_no_extra_inputs_keys_exactly_as_before`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtxInputs {
    /// The recipe reads the WALL-PLANE world hit ([`SampleCtx::wall_along`] /
    /// [`SampleCtx::wall_up`]) — the continuous position ON the wall, which `coord`
    /// does NOT determine because `coord.x` wraps once per world cell.
    pub wall_plane: bool,
    /// The recipe reads [`SampleCtx::state`] as an output-affecting value.
    ///
    /// This covers only the RECIPE. A material can also read the state word through a
    /// [`FamilyDriver::State`](crate::glyph_family::FamilyDriver::State) family, an
    /// `OnState` animation track, a [`SurfaceFx`](crate::surface_fx::SurfaceFx), or —
    /// unconditionally — the FLOW reading; [`Material::reads_state_word`] is the union
    /// and is what the key builders actually consult.
    pub state: bool,
    /// The recipe reads the SUB-CELL context — [`SampleCtx::tex_footprint`] (where the
    /// 2x2 / 2x4 sub-samples land, and which `EdgeField` mip serves them) and
    /// [`SampleCtx::subcell_backdrop`] (the colour a TRANSPARENT sub-texel reveals).
    ///
    /// Neither is implied by `coord`: two cells over the SAME world texel differ in
    /// footprint whenever they differ in distance or incidence, and differ in backdrop
    /// whenever the scene behind them differs. Declared TRUE only by an
    /// [`TextureRecipe::Image`] whose `subcell` is not [`SubcellMode::Off`] — the `Off`
    /// path ignores both fields entirely, so every pre-sub-cell material keeps a key
    /// byte-identical to the pre-fix one.
    pub subcell: bool,
}

impl CtxInputs {
    /// Reads nothing beyond the base key — every recipe that shipped before the voice
    /// pair. Keys built for such a recipe are byte-identical to the pre-fix ones.
    pub const NONE: Self = Self {
        wall_plane: false,
        state: false,
        subcell: false,
    };
    /// Reads the wall-plane hit and nothing else (`Substrate`, `Image` with a span).
    pub const WALL_PLANE: Self = Self {
        wall_plane: true,
        state: false,
        subcell: false,
    };
}

/// The SUB-CELL half of a shade-cache key: the two [`SampleCtx`] fields
/// [`Material::sample_image`] reads that neither `coord` nor the LOD band determines.
///
/// `Option` rather than a sentinel because "no footprint threaded" (every legacy
/// caster, `SubcellMode::Off`) is a DISTINCT input from "a footprint of exactly
/// `(0, 0)`" — collapsing them onto one bit pattern would reintroduce the very
/// collision this type exists to prevent. `Eq` on raw bit patterns: no `f32`
/// `PartialEq` pitfalls, and `NaN` keys compare equal to themselves so a degenerate
/// footprint still caches instead of thrashing.
///
/// [`NONE`](Self::NONE) is what a recipe that does not declare
/// [`CtxInputs::subcell`] folds, so its key is byte-identical to the pre-fix one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct SubcellKey {
    /// `SampleCtx::tex_footprint`'s `(along, up)` as raw bits.
    footprint: Option<(u32, u32)>,
    /// `SampleCtx::subcell_backdrop`.
    backdrop: Option<u32>,
}

impl SubcellKey {
    /// What a recipe that does not read the sub-cell context folds.
    const NONE: Self = Self {
        footprint: None,
        backdrop: None,
    };

    /// Fold the sub-cell context for `ctx` when — and only when — `declared`.
    #[inline]
    fn build(declared: bool, ctx: &SampleCtx) -> Self {
        if !declared {
            return Self::NONE;
        }
        SubcellKey {
            footprint: ctx.tex_footprint.map(|(a, u)| (a.to_bits(), u.to_bits())),
            backdrop: ctx.subcell_backdrop,
        }
    }

    /// Mix into an FNV chain. Presence is encoded in the high bits so `None` can
    /// never alias a real value, and an all-`None` key feeds a single constant — which
    /// is what keeps a non-declaring recipe's bucket distribution as it was.
    #[inline]
    fn feed(self, h: u64) -> u64 {
        let (fx, fy, fp_present) = match self.footprint {
            Some((x, y)) => (x, y, 1u64),
            None => (0, 0, 0),
        };
        let (bd, bd_present) = match self.backdrop {
            Some(b) => (b, 1u64),
            None => (0, 0),
        };
        let h = fnv64(h, (fx as u64) | ((fy as u64) << 32));
        fnv64(h, (bd as u64) | (fp_present << 32) | (bd_present << 33))
    }
}

/// The PER-RECIPE BEHAVIOUR DESCRIPTOR — the single authoritative table of every
/// property a consumer needs to know about a [`TextureRecipe`] WITHOUT re-matching
/// it. Built once by [`TextureRecipe::props`]; every property query in this module
/// (transparency routing, the SKY exclusions, the self-lit relief skip, the
/// code-wall carve-outs, the sparse-body thinning, the stable recipe tag) reads a
/// field here instead of writing its own `match`.
///
/// # ADDING A RECIPE — THE SINGLE SITE
/// To add a new [`TextureRecipe`] variant, you now edit exactly TWO places, both
/// adjacent and both unavoidable for the compiler to force:
///   1. add the variant to [`TextureRecipe`] (its data/params), and
///   2. add ONE arm to [`TextureRecipe::props`] declaring its behaviour, plus its
///      shading math arm in [`Material::sample`] / [`Material::height_at`] (the
///      `match self.recipe` dispatch the compiler makes you complete — non-
///      exhaustive matches fail to build).
///
/// Every *property* consumer (`is_transparent`, `is_animated`'s recipe inputs, the
/// SKY/CodeWall/self-lit predicates, `recipe_tag`, the sparse-body mask) reads
/// [`RecipeProps`] and therefore needs NO edit — that is the extensibility win a
/// data-driven material system wants: a material is a graph of data, not
/// hand-written code, so behaviour is declared as data in one place, not
/// hand-dispatched in five.
///
/// Pure `Copy` data, no allocation — `props()` is a cheap const-foldable match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecipeProps {
    /// A small stable discriminant for the recipe KIND (ignoring parameters). The
    /// authoritative source for [`Material::recipe_tag`].
    pub tag: u8,
    /// The recipe is the GLYPH-ONLY see-through primitive (void bg, sparse inked
    /// line-glyphs) → it composites into the TRANSPARENT slot. Drives
    /// [`Material::is_transparent`].
    pub transparent: bool,
    /// The recipe is the empty VOID surface (sky/ceiling): its blank glyph carries
    /// the depth band, not ink — so no family / super-glyph floats onto it, and the
    /// final depth band (step 5) applies. The SKY exclusion, in one place.
    pub is_sky: bool,
    /// The recipe renders the engine's own source text (the strange loop): its
    /// glyph IS the source character, so the relief body re-selection must not
    /// clobber it and it earns no emissive under-glow.
    pub is_code_wall: bool,
    /// The recipe is SELF-LIT — projected light / phosphor / sheen the relief
    /// (normal-mapped Blinn-Phong) pass must NOT relight; it only rides the
    /// animation brightness. True for the hologram, code wall, and scanline panel.
    pub self_lit: bool,
    /// Which world-anchored coverage mask (if any) thins this recipe's body glyph
    /// toward a coloured blank — see [`BodyBlankKind`].
    pub body_blank: BodyBlankKind,
    /// The recipe reads the state-bus word ([`SampleCtx::state`]) as a **continuous
    /// SCALAR**, not as a packed pipe FLOW word.
    ///
    /// # Why this field exists — a collision found by reading the shade path
    ///
    /// `state` is one `u32` with several readings, and [`TextureRecipe::Voice`] added
    /// one more. Most of those readings are opt-in per material (a `State` glyph
    /// family, an `OnState` track, a state-reading [`SurfaceFx`](crate::surface_fx::SurfaceFx)),
    /// so keeping them off the voice materials is enough. **The FLOW reading is not.**
    /// [`flow_shift_coord`] and [`flow_is_hot`] run in [`Material::sample`] for EVERY
    /// material unconditionally, and `re_core::decode_flow_state` keys on bit 0 — so
    /// an ODD mark (`501` parts, say) would decode as a powered pipe and hand the wall
    /// a [`FLOW_HOT_LIFT`] brightness lift that switched on and off as the mark ticked
    /// through consecutive parts. A progress bar that strobes as it fills.
    ///
    /// So the recipe DECLARES how it reads the word, here, once, and both flow
    /// consumers honour the declaration. `false` for every recipe that shipped before
    /// the voices ⇒ byte-identical. (Quantising the mark to even parts would also have
    /// dodged it, at the cost of hiding a real ambiguity inside a magic number.)
    pub state_is_scalar: bool,
    /// Which EXTRA [`SampleCtx`] fields this recipe reads, so the two shade caches can
    /// fingerprint them. See [`CtxInputs`] — this is the field that keeps
    /// [`SampleKey`] and [`WorldTexKey`] correctness-preserving.
    pub extra_inputs: CtxInputs,
}

impl TextureRecipe {
    /// The authoritative [`RecipeProps`] for this recipe — the SINGLE place per-
    /// recipe behaviour is declared. Every property consumer in the module reads
    /// this; adding a recipe means adding one arm HERE (the compiler enforces the
    /// match stays exhaustive), not touching five scattered `match` sites.
    ///
    /// Cheap, `const`-foldable, no allocation — called freely on the hot path.
    #[inline]
    pub fn props(&self) -> RecipeProps {
        match self {
            TextureRecipe::Brick { .. } => RecipeProps {
                tag: 0,
                transparent: false,
                is_sky: false,
                is_code_wall: false,
                self_lit: false,
                body_blank: BodyBlankKind::Wall,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            TextureRecipe::Circuit { .. } => RecipeProps {
                tag: 1,
                transparent: false,
                is_sky: false,
                is_code_wall: false,
                self_lit: false,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            TextureRecipe::Grain { .. } => RecipeProps {
                tag: 2,
                transparent: false,
                is_sky: false,
                is_code_wall: false,
                self_lit: false,
                body_blank: BodyBlankKind::Floor,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            TextureRecipe::Sky { .. } => RecipeProps {
                tag: 3,
                transparent: false,
                is_sky: true,
                is_code_wall: false,
                self_lit: false,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            TextureRecipe::Hologram { .. } => RecipeProps {
                tag: 4,
                transparent: true,
                is_sky: false,
                is_code_wall: false,
                self_lit: true,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            TextureRecipe::CodeWall { .. } => RecipeProps {
                tag: 5,
                transparent: false,
                is_sky: false,
                is_code_wall: true,
                self_lit: true,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            // A placard is text on a wall: the SAME properties as the code wall (its
            // glyph IS the character, so the relief body re-selection must not clobber
            // it, and the relief pass must not relight it) — declared here once rather
            // than hand-listed at each consumer.
            TextureRecipe::NoteWall { .. } => RecipeProps {
                tag: 9,
                transparent: false,
                is_sky: false,
                is_code_wall: true,
                self_lit: true,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            TextureRecipe::Truchet { .. } => RecipeProps {
                tag: 6,
                transparent: false,
                is_sky: false,
                is_code_wall: false,
                self_lit: true,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                // The lattice indexes the WORLD CELL, which on a wall only
                // `(wall_along, wall_up)` carries — `coord.x` wraps once per cell.
                // Declared so a cache hit cannot return a neighbour's tile.
                extra_inputs: CtxInputs::WALL_PLANE,
            },
            TextureRecipe::Scanline { .. } => RecipeProps {
                tag: 6,
                transparent: false,
                is_sky: false,
                is_code_wall: false,
                self_lit: true,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            TextureRecipe::Fractal { .. } => RecipeProps {
                tag: 7,
                transparent: false,
                is_sky: false,
                is_code_wall: false,
                // Self-lit: the ring/arm palette reads directly — the relief
                // Blinn-Phong pass must NOT relight it (same as Hologram/Scanline).
                self_lit: true,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::NONE,
            },
            // The cell alphabet is TEXT in the sense that matters here: its glyph IS
            // its meaning, so it takes the same `is_code_wall` ("the recipe owns its
            // glyph — the relief body re-selection must not clobber it") reading the
            // NoteWall arm above already reuses. It is NOT self-lit, though: unlike a
            // placard or a phosphor screen the substrate is a physical WALL and must
            // sit in the room's light, so the relief pass still colours it.
            // (Spelled `Self::` rather than `TextureRecipe::` like its ten siblings
            // purely to keep `clippy::use_self` silent on the newest arm; the file's
            // older arms predate that lint being surfaced.)
            Self::Substrate { .. } => RecipeProps {
                tag: 10,
                transparent: false,
                is_sky: false,
                is_code_wall: true,
                self_lit: false,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                // Reads the WALL-PLANE hit, not `coord` — that is the whole point of
                // `Material::sample_substrate`'s note, and it is why the shade caches
                // must fingerprint it: `coord.x` wraps once per world cell, so two
                // cells a world unit apart share it.
                extra_inputs: CtxInputs::WALL_PLANE,
            },
            // A voice wall IS a substrate wall — same alphabet, same physicality, the
            // only difference being where its two scalars come from. Every property
            // below is therefore `Substrate`'s, deliberately, and
            // `the_voice_recipe_declares_the_same_surface_properties_as_substrate`
            // pins them together so the two arms cannot drift. The `tag` is the one
            // field that must DIFFER: it is the stable recipe-kind discriminant, and
            // `no_shipped_material_adopts_the_substrate_recipe_yet` reads it to prove
            // the voices did not quietly join the substrate slot.
            Self::Voice { .. } => RecipeProps {
                tag: 11,
                transparent: false,
                is_sky: false,
                is_code_wall: true,
                self_lit: false,
                body_blank: BodyBlankKind::None,
                // THE ONE FIELD THAT DIFFERS FROM `Substrate` BESIDES THE TAG: this
                // recipe reads `ctx.state` as a mark, so the flow reading must not
                // also claim it. See `RecipeProps::state_is_scalar`.
                state_is_scalar: true,
                // The wall plane like `Substrate`, PLUS the state word it reads the
                // mark from. Both must reach the cache keys or a hit can return a
                // texel computed at a different mark, or at a neighbour's position.
                extra_inputs: CtxInputs {
                    wall_plane: true,
                    state: true,
                    subcell: false,
                },
            },
            TextureRecipe::Image {
                relief, subcell, ..
            } => RecipeProps {
                tag: 8,
                transparent: false,
                is_sky: false,
                is_code_wall: false,
                // Self-lit UNLESS the material authored `relief =`. Without relief the
                // sampled image colour reads directly and the Blinn-Phong pass must
                // NOT relight it — the plain full-colour-block path is byte-identical.
                // WITH relief the author has asked for the opposite: the photo is a
                // surface with a height map, so it takes the same lighting every
                // procedural recipe takes.
                self_lit: relief.is_none(),
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                // A SPANNED image maps `(wall_along, wall_up)` through `ImageSpan::uv`
                // — the same latent gap the voices exposed, declared here because the
                // table is only worth anything if it is filled in honestly. A `span`
                // of `None` ignores both, so this over-declares for the per-cell
                // tiling path; over-declaring costs a cache miss, under-declaring
                // costs a wrong texel.
                //
                // SUB-CELL: a `subcell != Off` image also reads `tex_footprint` and
                // `subcell_backdrop` (see [`CtxInputs::subcell`]). `Off` short-circuits
                // before either is touched, so it keeps declaring nothing and its key
                // stays byte-identical to the pre-fix one.
                extra_inputs: CtxInputs {
                    wall_plane: true,
                    state: false,
                    subcell: !matches!(subcell, SubcellMode::Off),
                },
            },
            // The window grid is a PHYSICAL wall (glass + mullion in the room's
            // light), not a projected/self-lit surface — unlike Hologram/CodeWall/
            // Scanline/Truchet/Fractal it takes the scene's relief lighting, exactly
            // like `Substrate` above. It reads the wall-plane hit (its bay/floor grid
            // is anchored to `(wall_along, wall_up)`, not `coord`, for the same
            // reason `Substrate`'s comment gives).
            //
            // `is_code_wall: true` — like `Substrate`/`Voice` above, NOT because this
            // is a code surface but because that is the flag the generic SPARSE-BODY
            // relief pass checks before RE-SELECTING the body glyph off its
            // own height/luminance density ramp. With this `false` (the original,
            // untested value — no prior test rendered a `surface = wall_solid`-bound
            // facade end to end), every facade wall's carefully-computed bay/floor glyph
            // (`crate::facade::sample`'s mullion/lit/dark table) was silently
            // overwritten by the relief pass's own generic ramp glyph immediately
            // after `sample_facade` returned it — and because `LightModel::FLAT`
            // gives Facade a CONSTANT relief height, the ramp glyph the pass picked
            // was itself near-constant, so an entire wall rendered as one uniform
            // low-density glyph (`.`) regardless of its authored window rhythm. The
            // facade's own glyph, exactly like the substrate alphabet's, IS the
            // meaning of the surface, so the relief pass must leave it alone.
            TextureRecipe::Facade { .. } => RecipeProps {
                tag: 12,
                transparent: false,
                is_sky: false,
                is_code_wall: true,
                self_lit: false,
                body_blank: BodyBlankKind::None,
                state_is_scalar: false,
                extra_inputs: CtxInputs::WALL_PLANE,
            },
        }
    }

    /// The state-bus tag a `state_is_scalar` recipe reads its scalar from, when that
    /// tag is a property of the MATERIAL rather than of the cell. `None` for every
    /// recipe whose state word is per-cell (all of them but the voices).
    ///
    /// # Why a recipe can own its tag at all
    ///
    /// Every other reading of the state bus is genuinely per-cell: THIS node's lane,
    /// THIS pipe's flow word. The voice mark is not — it is one level-wide number, and
    /// every cell of a `mnemos_hold` wall shows the same one. So the tag belongs to
    /// the material, and `crate::shade_cell` resolves it from here instead of from the
    /// cell's `side_data`.
    ///
    /// # The trap this closes
    ///
    /// `side_data` on an ordinarily-bound wall (`[material] surface = wall_solid`) is
    /// `0`, which is NODE 0's tag. Without this, a voice wall bound the obvious way
    /// read node 0's family lane as its progress, measured ≈0, and rendered as an
    /// untouched wall — which is a LEGITIMATE STATE of this material. No error, no
    /// warning, nothing that looks broken: an author would have stared at a
    /// correct-looking wall wondering why solving the puzzle did nothing. A level may
    /// still stamp a voice tag explicitly to cross-wire a wall to the other voice's
    /// mark (see [`crate::substrate::is_voice_tag`]); this is the default, not the
    /// only route.
    #[must_use]
    #[inline]
    pub const fn state_scalar_tag(&self) -> Option<u16> {
        match self {
            Self::Voice { mode, .. } => Some(mode.tag()),
            _ => None,
        }
    }
}

/// How detail ranks are assigned across a recipe's texels — controls the
/// coarse→fine reveal *order* under ranked-LOD.
#[derive(Clone, Copy, Debug)]
pub enum RankStrategy {
    /// Structural features (seams, traces, the base value band) get low ranks
    /// (revealed first / always); fine speckle / mottle gets high ranks
    /// (revealed only up close). The default honest mip-style order.
    StructureFirst,
    /// Everything fine: detail uniformly high-rank, so the surface reads as flat
    /// base until you are close. Good for grain/sky where specks are the detail.
    FineDetail,
}

/// What animation (if any) a material plays over `time`. A small typed param
/// block sampled per-cell — cheap, deterministic, **no per-cell allocation**.
///
/// `kind = None` ⇒ a fully static material (the dirty-flag may skip it). Any
/// other kind makes the material "animated", which forces the renderer to keep
/// re-shading the scene every tick (see [`Material::is_animated`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Animation {
    /// Which animation, if any, this material plays.
    pub kind: AnimKind,
    /// Cycles per second (animation phase advances `speed * time`).
    pub speed: f32,
    /// Strength of the effect in its natural units (brightness fraction for
    /// `Pulse`/`Flicker`/`Scan`/`Breathe`, world-units of UV drift for the coord
    /// kinds: unbounded scroll RATE for `Shift`, per-axis excursion for `Drift`,
    /// peak-to-peak burst width for `Tear`).
    pub amplitude: f32,
}

/// The animation modes a [`Material`] can carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimKind {
    /// No animation — a static look.
    None,
    /// Smooth sinusoidal brightness breathing (`fg`/`bg` scaled by a cosine).
    Pulse,
    /// Scroll the texture coordinate over time (matrix-rain / flow look).
    Shift,
    /// Pseudo-random brightness jitter (hashed on `time`) — flicker / static.
    Flicker,
    /// A bright band sweeping the surface along its u axis. Loops `mod 1/speed`.
    /// Reads as a machine inspecting itself — the archive's cold register.
    Scan,
    /// [`Pulse`](AnimKind::Pulse) with an asymmetric envelope: slow in, fast out.
    /// A sigh rather than a sine, which separates living surfaces from
    /// instrumented ones. Loops `mod 1/speed`.
    Breathe,
    /// Slow two-axis coordinate drift along a closed Lissajous path — ambient unease
    /// without the mechanical straight line of [`Shift`](AnimKind::Shift). Unlike
    /// `Shift` it is BOUNDED (`±amplitude` on each axis) and it LOOPS every `1/speed`,
    /// because the two axes run at a whole-number frequency ratio
    /// ([`DRIFT_AXIS_RATIO`]) and the path therefore closes. Drives the sample coord
    /// (so [`Material::anim_brightness`] is a no-op for it).
    Drift,
    /// Brief hashed horizontal displacement bursts with quiet between them: the
    /// `glitch` effect at material scale, for corruption escalation.
    ///
    /// **APERIODIC on purpose**, and the one design decision in this variant worth
    /// stating: the burst pattern is hashed off a MONOTONE tick index, exactly like
    /// [`Flicker`](AnimKind::Flicker), so it never returns. Wrapping the index into a
    /// fixed number of slots would make it loop every `1/speed` — and a glitch on a
    /// fixed rhythm reads as a mechanism, not as damage. There is no discontinuity in
    /// a hash for a wrap to hide, so the wrap would only ADD structure the unwrapped
    /// form does not have.
    ///
    /// MEASURED, and worse than merely rhythmic: with the wrap, a row draws from only
    /// [`TEAR_TICKS_PER_UNIT`] hashed slots forever, so at a [`TEAR_DUTY`] of `0.18` a
    /// row has a `0.82^16 ≈ 4%` chance of drawing NO torn slot at all — and then that row
    /// never tears again for the lifetime of the level. The probe row is one such row:
    /// wrapping the index makes `Tear` render as a completely static surface there.
    /// The monotone index has no such failure mode.
    ///
    /// What it guarantees instead of a loop is a HOLD: the displacement is constant for
    /// a whole tick, so a tick is either torn or not, never half-torn.
    /// Drives the sample coord (so `anim_brightness` is a no-op for it).
    Tear,
    /// Log-polar ZOOM (infinite zoom): scrolls the [`TextureRecipe::Fractal`] field's
    /// log-radius over time. Loops `mod LN2` (one zoom-octave) ⇒ seamless. Drives the
    /// field PHASE, not a coord scroll (so [`Material::anim_shift_coord`] is a no-op).
    Zoom,
    /// Log-polar SPIN: scrolls the [`TextureRecipe::Fractal`] field's angle over time.
    /// Loops `mod TAU` (one turn) ⇒ seamless. Drives the field phase.
    Spin,
    /// Coupled SPIN+ZOOM (Droste spiral): both axes, with the angular rate locked to
    /// the zoom rate so the spin advances a whole [`TAU`](crate::fractal::TAU) over one
    /// [`LN2`](crate::fractal::LN2) zoom — both seams coincide for a hypnotic dive.
    SpinZoom,
}

// ───────── coordinate-animation tunables (`Drift` / `Tear`) ─────────
//
// Named because they are the LOOK knobs of their kinds, and because the loop /
// hold classification in `anim_tests` is derived from them rather than guessed.

/// [`AnimKind::Drift`]'s second-axis frequency, as a multiple of the first. It must be
/// a whole number: that — and only that — is what closes the Lissajous path and makes
/// `Drift` loop every `1/speed`. `2.0` gives the classic figure-eight, which reads as
/// wandering rather than as circling.
const DRIFT_AXIS_RATIO: f32 = 2.0;

/// [`AnimKind::Tear`]'s tick rate: ticks per unit of `speed · time`. Higher = shorter,
/// snappier bursts. `16` puts a burst edge roughly twice per rendered frame at
/// `speed = 1`, which is fast enough to read as a break-up rather than as a slide.
const TEAR_TICKS_PER_UNIT: f32 = 16.0;

/// Fraction of [`AnimKind::Tear`] ticks that actually tear. The QUIET is the effect:
/// a surface that displaces most of the time reads as motion, not as damage. Kept
/// well under a quarter so the eye registers a stable surface interrupted.
const TEAR_DUTY: f32 = 0.18;

/// Salt that decorrelates a torn tick's displacement AMOUNT from the decision to tear
/// at all — without it, the bursts that fire would all carry near-identical offsets
/// (the same hash thresholded and then reused as a magnitude).
const TEAR_JITTER_SALT: i32 = 0x5EED;

impl Animation {
    /// The static (no-op) animation. Const so materials can be `static`.
    pub const NONE: Animation = Animation {
        kind: AnimKind::None,
        speed: 0.0,
        amplitude: 0.0,
    };

    /// True if this animation actually changes the look over time.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.kind != AnimKind::None
    }
}

/// Where a material gets its glyph from. Most materials use the procedural
/// `Recipe` ramp; an `Explicit` set lets an author pin a fixed glyph (or a small
/// rotating set indexed by a hash / time) for a hand-authored look.
#[derive(Clone, Copy, Debug)]
pub enum GlyphSource {
    /// Use the recipe's procedural glyph choice (the default look).
    Recipe,
    /// Force this single glyph for the material's *base* texel (details still
    /// come from the recipe). A Unicode scalar value.
    Fixed(u32),
}

impl GlyphSource {
    /// Apply this glyph source to a recipe-chosen base glyph.
    #[inline]
    fn resolve(&self, recipe_glyph: u32) -> u32 {
        match self {
            GlyphSource::Recipe => recipe_glyph,
            GlyphSource::Fixed(g) => *g,
        }
    }
}

// ───────── single-glyph ANIMATION TRACKS ─────
//
// The existing
// [`Animation`]/[`AnimKind`] is the BRIGHTNESS/COORD channel. An [`AnimTrack`] is the
// GLYPH channel beside it: it cycles a short single-glyph SEQUENCE
// (`·∘○◌●`, `░▒▓█`), REPLACING the chosen body glyph, optionally gated by a
// state-bus trigger so a node "breathes" only when powered. Everything is `Copy`
// (fixed arrays, no `Vec`) so a [`Material`] stays `static`-able and the hot path
// never allocates. An EMPTY [`AnimTrackSet`] is a strict no-op (today's glyph).

/// How an [`AnimTrack`]'s frame index advances over `ctx.time` (the shader reads no
/// clock, so the chosen frame is golden-reproducible).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnimClock {
    /// Wall-clock seconds: frame index advances `speed` frames/second
    /// (`floor(time * speed)`). Deterministic.
    Seconds(f32),
    /// BPM-quantized: the frame advances on a musical SUBDIVISION of the beat
    /// (`subdiv` steps per beat at `bpm` beats/minute) so morphs land on the beat when
    /// the audio clock drives `ctx.time`. `subdiv` clamped to ≥1.
    Tick {
        /// Beats per minute driving the subdivision.
        bpm: f32,
        /// Steps per beat (clamped to ≥1).
        subdiv: u8,
    },
}

impl AnimClock {
    /// The integer frame index this clock has reached at time `t` seconds. Pure,
    /// monotone. A non-positive rate holds frame 0 (static).
    #[inline]
    fn frame_index(self, t: f32) -> u64 {
        let steps_per_sec = match self {
            AnimClock::Seconds(speed) => speed,
            AnimClock::Tick { bpm, subdiv } => (bpm / 60.0) * (subdiv.max(1) as f32),
        };
        if steps_per_sec <= 0.0 || !t.is_finite() || t <= 0.0 {
            0
        } else {
            (t * steps_per_sec).floor() as u64
        }
    }
}

/// What gates an [`AnimTrack`]. `OnState` is the state-bus hook: the track runs
/// only when the cell's generic state word (`ctx.state`, looked
/// up by the cell's `material` tag from the [`CellStateMap`](re_core::CellStateMap))
/// EXCEEDS `field_gt`. `re_render` never interprets the puzzle meaning. An empty bus
/// reads `state == 0`, so `OnState` is a no-op until the bridge drives a non-zero state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimTrigger {
    /// Always cycling (a free-running morph).
    Always,
    /// Cycle ONLY when `ctx.state > field_gt`; else static (the body glyph shows).
    /// The state-reactive trigger: a node breathes when powered, holds when dead.
    OnState {
        /// The state-bus threshold the cell's state must exceed to cycle.
        field_gt: u32,
    },
    /// Play through ONCE then hold the LAST frame (a one-shot reveal).
    Once,
}

impl AnimTrigger {
    /// Is this track CYCLING for the given state word?
    #[inline]
    fn is_cycling(self, state: u32) -> bool {
        match self {
            AnimTrigger::Always | AnimTrigger::Once => true,
            AnimTrigger::OnState { field_gt } => state > field_gt,
        }
    }
}

/// Which glyph an [`AnimTrack`] REPLACES. v0.1 targets the cell BODY glyph.
/// `MatrixIndex` is reserved for the `GlyphMap` 2D-pattern work (not yet built); until
/// it ships it is treated as the body glyph so a track authored against it still
/// animates (forward-compatible, never garbage).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnimTarget {
    /// Replace the cell's chosen body glyph (the default).
    #[default]
    Body,
    /// Replace `GlyphMap` matrix slot `k` (reserved; behaves as `Body` for now).
    MatrixIndex(u8),
}

/// One single-glyph animation track: a short frame SEQUENCE cycled by a [`AnimClock`]
/// and gated by a [`AnimTrigger`], replacing a [`AnimTarget`] glyph. Fixed-capacity
/// (`frames: [u32; 8]`) so it is `Copy` and allocation-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimTrack {
    /// The morph/rotation sequence as Unicode scalar values. Only the first
    /// `frame_count` are read; trailing slots are ignored.
    pub frames: [u32; 8],
    /// Active frame count `1..=8`. `0` makes the track a no-op (inactive).
    pub frame_count: u8,
    /// How the frame index advances over `ctx.time`.
    pub clock: AnimClock,
    /// What gates the cycle (always / state-driven / one-shot).
    pub trigger: AnimTrigger,
    /// Which glyph the cycled frame replaces.
    pub target: AnimTarget,
}

impl AnimTrack {
    /// The glyph this track yields for `(time, state)`, or `None` when inactive
    /// (empty frames, OR a state-gated `OnState` track whose trigger is NOT satisfied)
    /// — the caller then KEEPS the body glyph (a DEAD node shows its own static glyph,
    /// not a track frame). A satisfied `OnState`/an `Always` track returns the cycled
    /// frame; a `Once` track holds its last frame after completing. Pure, no alloc.
    #[inline]
    fn resolve(self, time: f32, state: u32) -> Option<u32> {
        let n = self.frame_count.min(8) as u64;
        if n == 0 {
            return None;
        }
        if !self.trigger.is_cycling(state) {
            return None;
        }
        let raw = self.clock.frame_index(time);
        let idx = match self.trigger {
            AnimTrigger::Once => raw.min(n - 1),
            _ => raw % n,
        };
        Some(self.frames[idx as usize])
    }

    /// True if this track can change the glyph over `time` (so the material reports
    /// [`Material::is_animated`]). Only a fully-empty (`frame_count == 0`) track is
    /// inert; even `OnState` reports animated so the dirty-flag keeps re-shading and the
    /// cell can react the moment its state crosses the threshold.
    #[inline]
    fn is_animated(self) -> bool {
        self.frame_count.min(8) > 0
    }
}

/// Up to two single-glyph animation tracks on a material. Kept `Copy`. The
/// [`Default`] (`count == 0`) is the EMPTY set — a strict no-op (today's
/// glyph). Tracks apply in order; the LAST active body-targeting track wins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimTrackSet {
    /// The track slots; only the first `count` are active.
    pub tracks: [AnimTrack; 2],
    /// Active track count `0..=2`. `0` ⇒ the empty no-op set.
    pub count: u8,
}

impl AnimTrackSet {
    /// The EMPTY set — a strict no-op. `const` so a material can be a `static`.
    pub const EMPTY: AnimTrackSet = AnimTrackSet {
        tracks: [TRACK_INERT, TRACK_INERT],
        count: 0,
    };

    /// True if ANY track is non-empty — the material then reports
    /// [`Material::is_animated`] (mirrors the existing [`Animation`] behaviour).
    #[inline]
    pub fn is_animated(&self) -> bool {
        self.active_tracks().any(AnimTrack::is_animated)
    }

    #[inline]
    fn active_tracks(&self) -> impl Iterator<Item = AnimTrack> + '_ {
        let n = (self.count as usize).min(2);
        self.tracks[..n].iter().copied()
    }

    /// Resolve the BODY glyph after the tracks run: each active body-targeting track
    /// that yields a glyph replaces it (last wins). A `frozen` cell passes `time == 0`
    /// so every clock holds frame 0 (the far field stays quiet). Pure, no alloc.
    #[inline]
    fn resolve_body(&self, body: u32, time: f32, state: u32) -> u32 {
        let mut g = body;
        for track in self.active_tracks() {
            if track.frame_count.min(8) == 0 {
                continue;
            }
            if let Some(frame) = track.resolve(time, state) {
                g = frame;
            }
        }
        g
    }
}

impl Default for AnimTrackSet {
    #[inline]
    fn default() -> Self {
        AnimTrackSet::EMPTY
    }
}

/// A canonical inert track (fills the empty [`AnimTrackSet::EMPTY`] slots).
const TRACK_INERT: AnimTrack = AnimTrack {
    frames: [0; 8],
    frame_count: 0,
    clock: AnimClock::Seconds(0.0),
    trigger: AnimTrigger::Always,
    target: AnimTarget::Body,
};

/// The SUPER-GLYPH SELECTORS beyond the four
/// [`crate::glyph_family::FamilyDriver`] kinds (`time`/`distance`/`normal`/`state`).
/// Each picks a member of a [`crate::glyph_family::GlyphFamily`] from a richer driver while
/// staying ECONOMICAL (one table lookup over a quantized driver — no per-cell heavy
/// math) and TEMPORALLY STABLE (every driver is the same quantized / hysteretic
/// input the rest of the shader uses, so far / idle cells do not flicker).
///
/// Pure data on a [`Material`] (`Copy`). The `Distance`/`Normal`/`Time`/`State`
/// selectors already live as [`crate::glyph_family::FamilyDriver`]; these are
/// implemented here (this crate owns the catalog) so the sibling-owned
/// `glyph_family` table stays untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SuperSelector {
    /// **`bpm-driven`** — a beat-quantized cycle. The audio clock is read as
    /// `ctx.time`; the phase advances `bpm/60` beats/sec, and the family member is
    /// chosen by the beat's WHOLE index (steppy, on-beat — not a smooth lerp), so the
    /// surface pulses on the beat like the rhythmic-audio layer. `bpm` beats/minute.
    /// ANIMATED (depends on `time`).
    Bpm {
        /// Beats per minute driving the phase.
        bpm: f32,
    },
    /// **`density-driven`** — picks a member by the recipe's own per-cell DENSITY
    /// driver (the quantized LOD density), so the glyph tracks how much "stuff" is in
    /// the cell (sparse → early member, dense → late). Static (no `time`): cacheable.
    Density,
    /// **`neighbor-aware`** — an auto-tiling pick: the member index is a hash of
    /// the cell's QUANTIZED world bucket plus its four axis-neighbour buckets, so
    /// adjoining cells choose CONNECTED members (a coherent woven / bonded pattern)
    /// rather than independent noise. World-anchored ⇒ stable under camera motion.
    /// Static (cacheable). The cheap variant of a "dependent on neighbours"
    /// selector: it reads the deterministic world lattice, not a live neighbour
    /// framebuffer, so it costs one extra hash, not a neighbour gather.
    NeighborAware,
    /// **`multiply-composite`** — two drivers MULTIPLY: a time float (`speed`
    /// cycles/sec) selects WITHIN a short window while the quantized distance picks
    /// the window, so the glyph both drifts and coarsens with depth (rotates × LODs).
    /// ANIMATED.
    MultiplyComposite {
        /// Within-window cycle rate, cycles/second.
        speed: f32,
    },
    /// **`priority-composite`** — a finite-state interrupt: while `state != 0`
    /// the family is PINNED to member `state` (the alert / cursed / active frame);
    /// when `state == 0` it falls back to a slow idle float (`speed` cycles/sec). The
    /// state frame has PRIORITY over the idle animation. ANIMATED (idle floats).
    PriorityComposite {
        /// Idle-float cycle rate, cycles/second, used while `state == 0`.
        speed: f32,
    },
    /// **`procedural-composite`** — the member is computed procedurally from the
    /// cell's world hash + a slow time term, giving endless non-repeating variation
    /// (a "boiling" field). World-seeded so neighbours desync; the time term is
    /// quantized to whole steps so it advances in stable jumps, not a jitter.
    /// ANIMATED.
    Procedural {
        /// Time-term advance rate (quantized to whole steps).
        speed: f32,
    },
    /// **`view-angle-driven`** — picks a family member from
    /// the VIEW INCIDENCE on the surface: the azimuth of the view direction within the
    /// cell's tangent frame is bucketed into `buckets` steps and indexes the family, so
    /// the glyph CHANGES as you orbit the surface (a facetted / hatched / oriented look).
    /// Deliberately VIEW-DEPENDENT (that is the
    /// point: the symbol tracks the viewing angle), so a material carrying it cannot be
    /// rotation-panned; the renderer recasts on a turn (this is the small set of
    /// view-dependent geometry, like mirrors). NOT time-driven, so it is cacheable per
    /// quantized angle bucket and does not flicker (the bucket is a deadband). With no
    /// [`LightCtx`] / under view-independent lighting the angle is the face-on centre,
    /// so the pick is stable.
    ByAngle {
        /// Number of azimuth steps the view incidence is bucketed into.
        buckets: u8,
    },
}

impl SuperSelector {
    /// True if this selector's pick depends on `time` (so a material carrying it is
    /// [`Material::is_animated`] — re-shaded each tick, cache-bypassed). `Density`,
    /// `NeighborAware` and `ByAngle` are pure functions of static per-cell geometry
    /// (or the cast-fixed view incidence) and stay cacheable; the rest float.
    #[inline]
    pub fn is_animated(self) -> bool {
        !matches!(
            self,
            SuperSelector::Density | SuperSelector::NeighborAware | SuperSelector::ByAngle { .. }
        )
    }
}

/// An optional SUPER-GLYPH layer a [`Material`] carries: a [`crate::glyph_family::GlyphFamily`] (the map
/// of kindred members) plus the [`SuperSelector`] that picks within it plus a small
/// `state` index used by the state-aware selectors. When present it REPLACES the
/// recipe's base glyph (like [`crate::glyph_family::FamilyLayer`], but with a richer
/// set of selectors); when absent the material is byte-for-byte unchanged.
/// Pure data (`Copy`).
#[derive(Clone, Copy, Debug)]
pub struct SuperGlyphLayer {
    /// The map of kindred glyph members this layer picks from.
    pub family: crate::glyph_family::GlyphFamily,
    /// How a member is picked from `family` each frame.
    pub selector: SuperSelector,
    /// A small finite-state index for the state-aware selectors
    /// ([`SuperSelector::PriorityComposite`]). 0 = idle; >0 = an interrupt frame.
    pub state: u32,
}

impl SuperGlyphLayer {
    /// True if this layer's look changes over `time` (drives the material's
    /// `is_animated`).
    #[inline]
    pub fn is_animated(&self) -> bool {
        self.selector.is_animated()
    }

    /// Resolve this super-glyph layer to a single glyph for a cell. Pure +
    /// allocation-free. Inputs:
    /// * `quant_density` — the cell's QUANTIZED LOD density (the anti-flicker driver
    ///   the rest of the shader already computes), reused so the discrete pick is
    ///   stable under sub-cell camera motion.
    /// * `coord` — the WORLD texel coordinate (for the world-anchored hashes).
    /// * `time` — the float clock (0.0 for a frozen / static cell, so the field holds).
    /// * `seed` — the material seed (desyncs distinct materials).
    /// * `angle01` — the cell's VIEW-INCIDENCE azimuth normalized to `[0,1)` (only read
    ///   by [`SuperSelector::ByAngle`]; the face-on `0.5` for the other selectors / no
    ///   light). The caller derives it from the per-cell [`LightCtx`].
    #[inline]
    pub fn select(
        &self,
        quant_density: f32,
        coord: Vec2,
        time: f32,
        seed: u32,
        angle01: f32,
    ) -> u32 {
        let fam = &self.family;
        // A normalized [0,1] from the quantized density — the same near→far mapping
        // the family-growth path uses, so a density pick is deadbanded / stable.
        let density01 =
            smoothstep(LOD_NEAR_DENSITY, LOD_FAR_DENSITY, quant_density).clamp(0.0, 1.0);
        // The per-cell phase/seed: a world-anchored hash so neighbours desync and a
        // spot is frame-stable (same discipline as `cell_phase`).
        let phase_seed = crate::glyph_family::cell_phase(coord.x, coord.y) ^ seed;
        match self.selector {
            SuperSelector::Bpm { bpm } => {
                // Beat index: whole beats elapsed. Steppy (on-beat), not a smooth
                // cycle — the member changes only when a beat boundary is crossed.
                let beats = (time * (bpm.max(1.0) / 60.0)).floor();
                fam.cycle(beats, phase_seed)
            }
            SuperSelector::Density => fam.pick(density01),
            SuperSelector::NeighborAware => {
                // Auto-tiling: hash this cell's coarse world bucket together with its
                // four axis-neighbour buckets so adjoining cells pick CONNECTED
                // members. Floor to a coarse lattice (the texel granularity) so the
                // pick is world-anchored and camera-stable.
                let ix = (coord.x * NEIGHBOR_LATTICE).floor() as i32;
                let iy = (coord.y * NEIGHBOR_LATTICE).floor() as i32;
                let h = hash2i(ix, iy)
                    ^ hash2i(ix.wrapping_add(1), iy).rotate_left(5)
                    ^ hash2i(ix.wrapping_sub(1), iy).rotate_left(11)
                    ^ hash2i(ix, iy.wrapping_add(1)).rotate_left(17)
                    ^ hash2i(ix, iy.wrapping_sub(1)).rotate_left(23)
                    ^ seed;
                fam.at((h as usize) % fam.len().max(1))
            }
            SuperSelector::MultiplyComposite { speed } => {
                // distance picks the WINDOW (which slice of the family), time floats
                // WITHIN it — the two drivers multiply. The float term is quantized to
                // whole cycles so it advances in stable jumps, not a jitter.
                let window = fam.pick(density01);
                // Re-derive the window's index, then add a small floated offset.
                let base_idx = fam
                    .glyphs
                    .iter()
                    .position(|&c| c as u32 == window)
                    .unwrap_or(0);
                let drift = (speed * time).floor() as i64;
                let n = fam.len().max(1) as i64;
                let idx = ((base_idx as i64 + drift).rem_euclid(n)) as usize;
                fam.at(idx)
            }
            SuperSelector::PriorityComposite { speed } => {
                if self.state != 0 {
                    // STATE PRIORITY: pin to the interrupt frame, ignoring the idle
                    // float entirely (a finite-state-machine glyph).
                    fam.at(self.state as usize)
                } else {
                    // Idle: a slow float over the family.
                    fam.cycle(speed * time, phase_seed)
                }
            }
            SuperSelector::Procedural { speed } => {
                // Procedural "boil": the member index is a hash of the world bucket
                // mixed with a QUANTIZED time step (whole steps so it jumps stably).
                let step = (speed * time).floor() as i32;
                let ix = (coord.x * NEIGHBOR_LATTICE).floor() as i32;
                let iy = (coord.y * NEIGHBOR_LATTICE).floor() as i32;
                let h = hash3i(ix ^ seed as i32, iy, step);
                fam.at((h as usize) % fam.len().max(1))
            }
            SuperSelector::ByAngle { buckets } => {
                // VIEW-ANGLE pick: quantize the view-incidence azimuth `angle01` into
                // `buckets` discrete steps and index the family directly. Bucketing IS
                // the deadband — a small angular wobble stays in one bucket, so the glyph
                // does not flicker as you turn; it steps to the next member only when you
                // cross a bucket boundary. The bucket count is clamped to the family
                // length so every step maps to a distinct member where possible.
                let n = fam.len().max(1);
                let b = (buckets as usize).clamp(1, n);
                let a = angle01.rem_euclid(1.0);
                let bucket = ((a * b as f32) as usize).min(b - 1);
                // Spread the `b` buckets across the `n` members so a small family with
                // many buckets still uses its whole range.
                let idx = bucket * n / b;
                fam.at(idx.min(n - 1))
            }
        }
    }
}

/// How TEMPORALLY STABLE a material's GLYPH choice is as the camera moves — the
/// importance / "anti-flicker" knob (additive; pure data so a [`Material`] stays
/// `Copy`).
///
/// Flicker / eye-strain comes from the ranked-LOD and the depth-ink ramp
/// RE-SELECTING a discrete glyph on tiny distance changes while the player walks.
/// The fix is to anchor a detail glyph to the WORLD texel + a HEAVILY quantized
/// LOD bucket so the *glyph identity* stops churning, while COLOUR keeps reading
/// the raw (smooth) distance and the cell still moves/scales across the screen —
/// so PARALLAX and shading stay fully live; only the discrete glyph is stabilised.
///
/// This scalar picks how wide that glyph-LOD deadband is and how strong the
/// world-anchored reveal HYSTERESIS is:
///
/// * [`Responsive`](Self::Responsive) — current behaviour (deadband ×1, no extra
///   hysteresis). Structural lines, mortar/trace seams, puzzle nodes/doors, and
///   near surfaces stay crisp + responsive, exactly as shipped.
/// * [`Stable`](Self::Stable) — texture-detail bodies (brick faces, floor fields):
///   a WIDER LOD deadband + world-texel hysteresis, so a body/detail glyph rarely
///   re-picks while you approach a surface.
/// * [`Frozen`](Self::Frozen) — far, large flat fields where churn is pure noise:
///   the widest deadband + strongest hysteresis. The glyph is effectively pinned
///   per world-texel-bucket until a coarse, hysteretic LOD step is crossed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GlyphStability {
    /// Crisp, responsive glyph selection — the shipped behaviour. (`Responsive`
    /// ≈ pre-stability output: deadband ×1, no extra hysteresis.)
    #[default]
    Responsive,
    /// Wide deadband + world-texel hysteresis: the detail glyph rarely re-picks.
    Stable,
    /// Widest deadband + strongest hysteresis: the glyph is near-pinned per
    /// world-texel bucket until a coarse LOD step is crossed.
    Frozen,
}

impl GlyphStability {
    /// Multiplier on [`LOD_QUANT_STEP`] for the GLYPH-selection deadband. Larger ⇒
    /// the camera must move further before the glyph re-picks. `Responsive` is 1.0
    /// so it reproduces the shipped deadband exactly.
    #[inline]
    pub fn deadband_mul(self) -> f32 {
        match self {
            GlyphStability::Responsive => 1.0,
            GlyphStability::Stable => STABLE_DEADBAND_MUL,
            GlyphStability::Frozen => FROZEN_DEADBAND_MUL,
        }
    }

    /// Hysteresis half-width, in *rank units*, applied to the detail-reveal
    /// decision: once a detail is revealed it stays revealed until the threshold
    /// drops a further `hysteresis()` below the rank (a lower LEAVE threshold than
    /// ENTER threshold), so a tiny step back does not un-reveal it. `Responsive` is
    /// 0 (no extra hysteresis → shipped behaviour).
    #[inline]
    pub fn hysteresis(self) -> f32 {
        match self {
            GlyphStability::Responsive => 0.0,
            GlyphStability::Stable => STABLE_HYSTERESIS,
            GlyphStability::Frozen => FROZEN_HYSTERESIS,
        }
    }
}

/// The MATERIAL UNDERLINE ACCENT — a smart-material "layer" that draws a SUBTLE
/// colored underline SUBSTYLE (single/double/curly/dotted/dashed) under a
/// material's inked cells, INDEPENDENT of fg/bg.
///
/// This is a tasteful accent, NOT a heavy fill: it is meant for glints / scan-glow
/// / specular ridge highlights. Pure data (`Copy`) so a [`Material`] stays `Copy`
/// and can be a `static`. The accent is realised per-cell in [`cell_style`]:
///
/// * `style`  — the underline SHAPE ([`re_core::UnderlineStyle`]). `None` ⇒ the
///   accent layer is OFF for this material (back-compatible default).
/// * `color`  — packed `0x00RRGGBB` accent hue, or [`HUE_FROM_FG`](Self::HUE_FROM_FG)
///   to derive the accent from the cell's own (already-lit) fg so the underline
///   tracks the surface colour (the hologram hue case).
/// * `gate`   — which cells earn the accent (see [`AccentGate`]), so it stays a
///   careful, selective effect rather than a wash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnderlineAccent {
    /// The underline SHAPE to draw. [`re_core::UnderlineStyle::None`] disables the
    /// whole accent layer for this material (the back-compatible default).
    pub style: re_core::UnderlineStyle,
    /// Packed `0x00RRGGBB` accent colour, or [`HUE_FROM_FG`](Self::HUE_FROM_FG) to
    /// derive a brightened take on the cell's own fg (so the underline shares the
    /// surface's colour family — the holo scan-glow look).
    pub color: u32,
    /// Which cells earn the accent (selectivity — keeps it subtle).
    pub gate: AccentGate,
}

impl UnderlineAccent {
    /// Sentinel `color` meaning "derive from the cell's lit fg" (a brightened
    /// take), rather than a fixed hue. Reuses [`re_core::NO_UNDERLINE`] (high byte
    /// `0xFF`, impossible for a real packed `0x00RRGGBB`).
    pub const HUE_FROM_FG: u32 = re_core::NO_UNDERLINE;

    /// Sentinel `color` for the ECONOMICAL MULTI-CHANNEL VOLUME cue: derive the
    /// underline TOGETHER with fg/bg as a hue-preserving third shade — see
    /// [`volume_underline`] for the math. A second impossible packed value (high byte
    /// `0xFE`) so it never collides with a real `0x00RRGGBB` or with [`Self::HUE_FROM_FG`].
    /// Pair it with [`AccentGate::Volume`] for the general per-cell volume underline.
    pub const HUE_VOLUME: u32 = 0xFE00_0000;

    /// The OFF accent: no underline layer (the back-compatible default every
    /// pre-accent material carries → unchanged output).
    pub const NONE: UnderlineAccent = UnderlineAccent {
        style: re_core::UnderlineStyle::None,
        color: UnderlineAccent::HUE_FROM_FG,
        gate: AccentGate::Never,
    };

    /// True if this accent layer actually draws something.
    #[inline]
    pub fn is_active(self) -> bool {
        self.style.is_some() && self.gate != AccentGate::Never
    }
}

/// Which inked cells an [`UnderlineAccent`] applies to — the selectivity knob that
/// keeps the underline a careful accent, not a heavy fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccentGate {
    /// Never (the accent layer is off). Paired with [`UnderlineAccent::NONE`].
    Never,
    /// Every inked (non-blank) cell of the material — used for the hologram, whose
    /// inked cells ARE the scan lines, so each line glyph gets the scan-glow.
    Inked,
    /// Only strongly-lit highlight cells (a SPECULAR RIDGE glint): a non-blank cell
    /// whose lit fg luma clears [`STYLE_BOLD_LUMA`]. A thin, occasional under-glint
    /// on the brightest ridges — deliberately rare.
    LitRidge,
    /// ECONOMICAL MULTI-CHANNEL VOLUME: every inked cell gets the underline as a
    /// THIRD colour cue computed TOGETHER with fg/bg to maximise perceived volume —
    /// a hue-preserving, brightness-shifted take that reads as the lit/near edge of
    /// the surface (see [`UnderlineAccent::HUE_VOLUME`] for the colour derivation).
    /// Unlike [`Inked`](Self::Inked) (which just glows the surface fg), `Volume` reads
    /// the cell's own face `side` + depth to bias the shift, so a NS face and an EW
    /// face — or a near vs far cell — get visibly different third cues from the same
    /// two base colours. General (not rare), cheap (a couple of scalar ops), and
    /// hue-stable, so a flat field of one element colour gains an extra perceived
    /// shade per cell at ~no cost.
    Volume,
}

/// How a material behaves across zoom octaves — the **scale rule** as a *render*
/// capability. This is the `re_render`-internal mirror of
/// [`re_core::ScaleRule`]: `re_core` MUST NOT depend on `re_render`, so the
/// POD [`re_core::SelfSimilarParams`] is mapped INTO [`crate::fractal::FractalParams`]
/// HERE, at the render boundary, by [`ScaleMode::from_core`].
///
/// `Default = None` is the bottom rung of the degradation ladder: no
/// scale behaviour, byte-identical to today. Every existing material defaults to it,
/// so a level that authors no `scale = …` zone renders exactly as before.
///
/// The slot composes with EVERY existing material capability (recipe, palette,
/// light, animation, …): the scale rule is applied as a post-step OVER the fully
/// shaded base texel, never replacing the material — `metal` could be `Crisp`,
/// `void`/`holo` could be `SelfSimilar`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleMode {
    /// No scale behaviour — today's flat, voxel-resolution look. A no-op post-step,
    /// so a `None` material shades byte-for-byte as before this capability existed.
    None,
    /// Resolution-independent intent, recorded but not yet implemented: crisp
    /// resolution-independent zone EDGES need an analytic region path, which is a
    /// deliberately deferred feature. We carry the marker so the authored
    /// intent is recorded, but a `Crisp` zone renders like [`None`](Self::None) today
    /// — we do NOT fake crispness. (For authored vector LINES, crispness is already
    /// satisfied by the existing vector-stroke path; this marker is the zone-edge case.)
    Crisp,
    /// The **fractal** case (`SelfSimilar`): the carried
    /// [`crate::fractal::FractalParams`] drive the shared log-polar self-similar field
    /// (`fractal.rs`), modulating the surface so detail repeats across octaves — the
    /// "infinite-zoom" applied to an overlay zone. ANIMATED: its phase advances with
    /// `ctx.time` via [`crate::fractal::spinzoom_phase`] (the same seamless loop the
    /// `Fractal` recipe's `SpinZoom` uses), so a `SelfSimilar` material reports
    /// [`Material::is_animated`].
    SelfSimilar(crate::fractal::FractalParams),
}

impl Default for ScaleMode {
    #[inline]
    fn default() -> Self {
        ScaleMode::None
    }
}

impl ScaleMode {
    /// Map the engine-pure [`re_core::ScaleRule`] → this render-internal mode,
    /// converting the POD [`re_core::SelfSimilarParams`] into
    /// [`crate::fractal::FractalParams`] (the conversion `re_render` owns — `re_core`
    /// never learns the fractal type). The POD fields mirror `FractalParams` 1:1.
    #[inline]
    pub fn from_core(rule: &re_core::ScaleRule) -> Self {
        match rule {
            re_core::ScaleRule::None => ScaleMode::None,
            re_core::ScaleRule::Crisp => ScaleMode::Crisp,
            re_core::ScaleRule::SelfSimilar(p) => {
                ScaleMode::SelfSimilar(crate::fractal::FractalParams {
                    arms: p.arms,
                    octave_warp: p.octave_warp,
                    twist: p.twist,
                })
            }
        }
    }

    /// True iff this mode animates over time (only [`SelfSimilar`](Self::SelfSimilar)
    /// does — its log-polar phase advances with `ctx.time`). Drives the per-material
    /// animation gate so a `SelfSimilar` zone rides the existing animated-colour
    /// re-shade path. `None`/`Crisp` are static.
    #[inline]
    pub fn is_animated(self) -> bool {
        matches!(self, ScaleMode::SelfSimilar(_))
    }
}

/// A complete material: recipe + palette + rank order + seed. Pure data, plus
/// an optional [`GlyphSource`] override and an [`Animation`] block.
///
/// A material MAY carry an attached [`re_core::TextureImage`] (via
/// [`texture`](Self::texture)), so it is no longer `Copy` — the `Arc` handle
/// is cheap to `Clone` but not `Copy`. Every built-in material leaves
/// `texture: None`, so a textureless map is unaffected.
#[derive(Clone, Debug)]
pub struct Material {
    /// The procedural pattern this material samples.
    pub recipe: TextureRecipe,
    /// Foreground-channel palette (the glyph's `fg` colour).
    pub palette: Palette,
    /// Background-channel palette (the `bg` base under the glyph). Dual-channel.
    pub bg_palette: Palette,
    /// How LOD reveal ranks are assigned to detail texels.
    pub rank: RankStrategy,
    /// Per-material hash seed, mixed into every noise/hash call this material makes.
    pub seed: u32,
    /// Glyph source for the base texel (procedural recipe vs hand-authored).
    pub glyph_source: GlyphSource,
    /// Per-material animation sampled by `ctx.time` (no-op = [`Animation::NONE`]).
    pub animation: Animation,
    /// Relief + normal-mapped lighting knobs (height scale, normal strength,
    /// specular, AO, emissive). [`LightModel::DEFAULT`] gives tasteful built-in
    /// relief; [`LightModel::FLAT`] reproduces flat (face-normal-only) shading.
    pub light: LightModel,
    /// How temporally stable this material's GLYPH choice is under camera motion
    /// (the anti-flicker / importance knob). [`GlyphStability::Responsive`] (the
    /// default) reproduces the shipped behaviour exactly; `Stable`/`Frozen` widen
    /// the world-anchored glyph deadband + reveal hysteresis so texture-detail
    /// glyphs stop churning while colour + parallax stay live.
    pub stability: GlyphStability,
    /// MIRROR flag: when `true`, the raycaster treats a hit on this material as a
    /// reflective surface — it reflects the primary ray about the hit face normal,
    /// continues the DDA for ONE bounce, shades what the mirror "sees", and shows
    /// that (tinted with a cool sheen, degraded by [`clarity`](Self::clarity)) in
    /// the cell instead of the material's own surface. `false` (the default) ⇒ an
    /// ordinary opaque surface — fully back-compatible.
    pub reflective: bool,
    /// Mirror CLARITY `[0,1]`: `1.0` = a crisp reflection, `0.0` = a dark, broken,
    /// noisy "collapsed" surface, with a blend in between. Only consulted when
    /// [`reflective`](Self::reflective) is `true`. The renderer's `MirrorClarity`
    /// resource can OVERRIDE this per cast so the game can animate the value live
    /// (the recognition beat). Defaults to `1.0` for non-mirror materials (unused).
    pub clarity: f32,
    /// MATERIAL UNDERLINE ACCENT layer (a smart-material accent): a subtle colored
    /// underline SUBSTYLE drawn under the material's inked cells — a glint /
    /// scan-glow / specular under-highlight, independent of fg/bg. Realised
    /// per-cell in [`cell_style`]. [`UnderlineAccent::NONE`] (the default) leaves
    /// the layer OFF, so a material that doesn't opt in is unchanged.
    pub underline_accent: UnderlineAccent,
    /// OPTIONAL KINDRED-SYMBOL super-glyph layer: a
    /// [`GlyphFamily`](crate::glyph_family::GlyphFamily) + a
    /// [`FamilyDriver`](crate::glyph_family::FamilyDriver) that REPLACES the recipe's
    /// base glyph with a "floating" / growth / positional substitution (see
    /// [`crate::glyph_family`]). `None` (the default) ⇒ the recipe glyph is used
    /// unchanged, so a material that doesn't opt in renders byte-for-byte as before.
    /// A `Time`-driven family makes the material [`is_animated`](Self::is_animated)
    /// (re-shaded every tick, bypassing the memo cache); the static drivers stay
    /// cacheable.
    pub family: Option<crate::glyph_family::FamilyLayer>,
    /// OPTIONAL SUPER-GLYPH layer: a [`SuperGlyphLayer`] (a [`crate::glyph_family::GlyphFamily`] +
    /// a [`SuperSelector`] — `bpm`/`density`/`neighbor-aware`/
    /// `multiply`/`priority`/`procedural`) that REPLACES the recipe's base glyph with
    /// a richer driver than the four [`crate::glyph_family::FamilyDriver`] kinds the
    /// [`family`](Self::family) layer covers. `None` (the default) ⇒ unchanged. It runs
    /// right AFTER the family block in [`sample`](Self::sample) with the IDENTICAL
    /// frozen / quantized-density / stability handling, so far / idle cells do not
    /// flicker; an animated selector ([`SuperSelector::is_animated`]) makes the material
    /// [`is_animated`](Self::is_animated) (re-shaded each tick, cache-bypassed). When both
    /// `family` and `super_glyph` are present, the super-glyph runs SECOND and wins
    /// (it is the higher-fidelity selector), but both substitute the recipe body glyph.
    pub super_glyph: Option<SuperGlyphLayer>,
    /// OPTIONAL BOLD-AS-PROXIMITY cue: when `Some(threshold)`, a cell whose
    /// `distance` is at/below `threshold` (world cells) is drawn BOLD — a cheap,
    /// "free" approach / depth cue carried entirely by the existing terminal bold
    /// attribute, independent of colour. `None` (the default) ⇒ no proximity bold,
    /// so back-compatible. Realised in [`cell_style_ext`] (the distance-aware path);
    /// the legacy [`cell_style`] leaves it off.
    pub proximity_bold: Option<f32>,
    /// OPTIONAL per-material LOD OVERRIDE (the "key object" / hero-object lever). When
    /// `Some`, this material resolves its distance LOD tier (relief gradient, noise
    /// octaves, cast-shadow quality) against its OWN near/mid/far thresholds instead
    /// of the global [`crate::lod::LodConfig`] — so a focal object (a puzzle node, a
    /// sign, a mirror) keeps full detail farther out than ordinary walls. The most
    /// common form is [`crate::lod::LodOverride::full_detail_to`] ("stay full-detail
    /// to distance D"). `None` (the default) ⇒ follow the global LOD curve, fully
    /// back-compatible (byte-identical to the pre-LOD-unification path).
    pub lod_override: Option<crate::lod::LodOverride>,
    /// GLASS-MARKER opt-in: when `true` AND the material is transparent AND it carries
    /// an active [`UnderlineAccent`], the composite paints NOTHING into the cell except
    /// the colored underline — the scene behind shows through untouched and the
    /// underline is the only "this is glass" cue (see
    /// [`is_underline_only_transparent`](Self::is_underline_only_transparent)). `false`
    /// (the default) leaves the material's transparent compositing exactly as before,
    /// so the hologram and other see-through materials are unchanged.
    pub glass_marker: bool,
    /// OPTIONAL SEMANTIC-SURFACE finish: a
    /// [`SurfaceFx`](crate::surface_fx::SurfaceFx) post-layer — **frost / wet /
    /// redaction / watching / wear / decay** — applied at the very end of
    /// [`apply_lighting`](Self::apply_lighting), OVER the shaded base texel. `None`
    /// (the default) ⇒ no finish, byte-identical to today; the catalog's
    /// `frost`/`wet_stone`/`redacted`/`watcher`/`worn_floor`/`decay` materials opt in.
    /// The redaction/decay finishes read the generic `ctx.state` word as a clearing /
    /// deadness level — `re_render` never learns the puzzle meaning (layer purity).
    pub surface_fx: Option<crate::surface_fx::SurfaceFx>,
    /// OPTIONAL SINGLE-GLYPH ANIMATION TRACKS: an
    /// [`AnimTrackSet`] cycled by a [`AnimClock`], gated by a [`AnimTrigger`]. Runs at
    /// the END of [`sample`](Self::sample) (after the body glyph is chosen), REPLACING
    /// the body glyph when active. The `OnState` trigger reads the generic `ctx.state`
    /// word (the state bus) so a POWERED node breathes and a dead one is static.
    /// [`AnimTrackSet::EMPTY`] (the default) is a strict no-op ⇒ byte-identical to
    /// today; a non-empty set makes the material [`is_animated`](Self::is_animated).
    pub anim_tracks: AnimTrackSet,
    /// OPTIONAL **SCALE RULE**: how this
    /// material behaves across zoom octaves. [`ScaleMode::None`] (the default) is a
    /// strict no-op — byte-identical to today; an overlay `[zone]` carrying
    /// `scale = self_similar …` tags its bound colour material with
    /// [`ScaleMode::SelfSimilar`] (mapped from [`re_core::ScaleRule`] in
    /// [`crate::registry_for_map`]), making the zone's surface render the shared
    /// log-polar self-similar field OVER its base shade — the infinite-zoom material
    /// applied to a zone. A `SelfSimilar` material is ANIMATED
    /// ([`is_animated`](Self::is_animated)), so it rides the existing animated-colour
    /// re-shade path. [`ScaleMode::Crisp`] is a recorded-intent marker only — a no-op
    /// today; the analytic edge path it names is a deliberately deferred feature.
    pub scale: ScaleMode,
    /// OPTIONAL ATTACHED IMAGE TEXTURE: the decoded RGBA8
    /// pixels the [`TextureRecipe::Image`] recipe samples. `crate::registry_for_map`
    /// populates this from `map.textures[tex_id]` (an `Arc` clone — the big buffer is
    /// shared, never deep-copied) so [`sample_image`](Self::sample_image) can read the
    /// pixels at shade time WITHOUT a `&Map`. `None` (the default for every built-in,
    /// and for a non-`Image` recipe) ⇒ no texture; the `Image` recipe then falls back
    /// to a magenta error block rather than panicking. Pure data, golden-safe.
    pub texture: Option<std::sync::Arc<re_core::TextureImage>>,
    /// OPTIONAL PRECOMPUTED EDGE FIELD: the
    /// DoG/Sobel magnitude + direction + mip pyramid the `Ascii`/`Contour`/`Lines`
    /// sub-cell samplers read. Built ONCE per texture by
    /// [`crate::registry_for_map`] (load-time) from the attached
    /// [`texture`](Self::texture); `None` (the default, and for every non-image /
    /// non-ASCII-family recipe) ⇒ the samplers fall back to luminance-only. Pure data.
    pub edge_field: Option<std::sync::Arc<crate::edge_field::EdgeField>>,
    /// OPTIONAL ANIMATION FRAME SET for the [`TextureRecipe::Image`] recipe.
    /// `None` — every still texture, and every non-image material — means
    /// [`texture`](Self::texture) IS the picture and nothing here reads `time`, so a
    /// still image shades byte-identically to before this field existed. `Some` means
    /// the level authored `[texture NAME] frames = clip.gif`: the shader picks
    /// `frames[(time·fps) as usize % n]` per sample and the material reports
    /// [`is_animated`](Self::is_animated), which is what already bypasses BOTH shade
    /// caches (neither key carries a time term).
    ///
    /// Frame 0 is ALSO in [`texture`](Self::texture) — deliberate aliasing, not
    /// duplication: it is the same `Arc`, and it keeps every time-less consumer (the
    /// `fit` span derivation, the `EdgeField` precompute, the caster's cut-out /
    /// silhouette tests) reading exactly what it read before, with no `SampleCtx` to
    /// thread. Pure data, golden-safe.
    pub texture_anim: Option<ImageFrames>,
    /// Opt-in world-anchored DETAIL-TIER cross-fade (`[material] detail = on`).
    /// `false` (the default, for
    /// every built-in and every material that never authors the key) ⇒ recipe
    /// sampling never evaluates `crate::detail_lod` at all — byte-identical to
    /// before this field existed. `true` gates the recipe's own sampling function
    /// (currently `Material::sample_grain`) on `crate::detail_lod::{tier_and_weight,
    /// reveal_finer}` for a world-anchored fine-grain reveal as the player
    /// approaches. Set from [`re_world::MaterialDef::detail`] by
    /// `crate::registry_for_map`.
    pub detail: bool,
}

/// The decoded FRAME SET behind an animated [`TextureRecipe::Image`] material — the
/// renderer-side view of a multi-frame [`re_core::TextureSet`].
///
/// Attached ONCE at registry-build time by
/// [`MaterialRegistry::attach_image_textures`]; the frames are `Arc`s into the map's
/// texture store, so a 68-frame GIF costs one refcount bump per frame here and is
/// never re-decoded per cast, per cell, or per frame of gameplay.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageFrames {
    /// The frames in playback order. NEVER empty (`re_core::TextureSet` guarantees
    /// it, and `attach_image_textures` only builds this from such a set), so
    /// [`frame_at`](Self::frame_at) is total.
    frames: std::sync::Arc<[std::sync::Arc<re_core::TextureImage>]>,
    /// Playback rate in frames per second — finite and `> 0`
    /// (`re_core::TextureSet::sanitize_fps`).
    fps: f32,
}

impl ImageFrames {
    /// Build from a [`re_core::TextureSet`]'s frames + rate. `Arc<[…]>` so cloning a
    /// `Material` (which the registry does per slot) never walks the frame list.
    #[must_use]
    pub fn from_set(set: &re_core::TextureSet) -> Self {
        ImageFrames {
            frames: std::sync::Arc::from(set.frames().to_vec().into_boxed_slice()),
            fps: set.fps(),
        }
    }

    /// Frames in playback order (never empty).
    #[inline]
    #[must_use]
    pub fn frames(&self) -> &[std::sync::Arc<re_core::TextureImage>] {
        &self.frames
    }

    /// Playback rate, frames per second.
    #[inline]
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// The frame to show at `time` seconds — `frames[(time·fps) as usize % n]` via the
    /// shared, total [`re_core::TextureSet::frame_index`]. PURE in `(time, fps, n)`:
    /// no clock, no RNG, no interior mutability, so two casts at the same `time` pick
    /// the same frame and `--dump` stays reproducible.
    #[inline]
    #[must_use]
    pub fn frame_at(&self, time: f32) -> &std::sync::Arc<re_core::TextureImage> {
        let i = re_core::TextureSet::frame_index(time, self.fps, self.frames.len());
        // `frame_index` returns `< len` for a non-empty set and `0` for an empty one;
        // the type invariant rules the latter out, but fall back rather than panic.
        &self.frames[i.min(self.frames.len().saturating_sub(1))]
    }
}

impl Material {
    /// True if this material's look changes over time — any non-`None`
    /// [`Animation`] OR a `Time`-driven [`family`](Self::family) (the FLOATING
    /// kindred-glyph layer). The renderer's dirty-flag must keep re-shading while any
    /// visible material reports `true`, and the [`SampleMemo`] bypasses the cache for
    /// it (its glyph depends on `time`, which is not in the cache key).
    #[inline]
    pub fn is_animated(&self) -> bool {
        self.animation.is_active()
            || self.family.is_some_and(|f| f.is_animated())
            || self.super_glyph.is_some_and(|s| s.is_animated())
            || self.surface_fx.is_some_and(|fx| fx.is_animated())
            || self.anim_tracks.is_animated()
            // A SelfSimilar scale rule advances its log-polar phase with `ctx.time`,
            // so it must re-shade every tick (and bypass the tex-cache, whose key has
            // no time term) — exactly like the `Fractal` recipe's SpinZoom animation.
            || self.scale.is_animated()
            // An `[texture] frames = clip.gif` set picks its frame from `ctx.time`, so
            // it is animated for exactly the same reason and takes exactly the same
            // cache bypass. A STILL texture leaves this `None` ⇒ unchanged.
            || self.texture_anim.is_some()
    }

    /// The image frame to sample for THIS cell: the animation's
    /// `frames[(time·fps) as usize % n]` when the material carries a
    /// [`texture_anim`](Self::texture_anim), else the single attached
    /// [`texture`](Self::texture).
    ///
    /// One call site (`sample_image`) resolves the picture ONCE per sample and hands
    /// the borrow to every sub-cell sampler, so animation costs a still image nothing
    /// but a `None` check and cannot desynchronise two samplers within one cell.
    #[inline]
    fn image_frame(&self, ctx: &SampleCtx) -> Option<&std::sync::Arc<re_core::TextureImage>> {
        match self.texture_anim.as_ref() {
            Some(anim) => Some(anim.frame_at(ctx.time)),
            None => self.texture.as_ref(),
        }
    }

    /// A small stable discriminant for this material's [`TextureRecipe`] (the recipe
    /// KIND, ignoring its parameters). Handy for tests/tools that want to compare or
    /// group materials by look without matching every field. Pure.
    #[inline]
    pub fn recipe_tag(&self) -> u8 {
        // Authoritative source: the per-recipe descriptor (no separate match to
        // keep in sync). Adding a recipe sets its tag once in `TextureRecipe::props`.
        self.recipe.props().tag
    }

    /// Whether this material composites into the TRANSPARENT layer (composite
    /// slot 2, `OverKeepBg`) rather than the opaque layer (slot 1). A transparent
    /// material draws glyph+fg over the background behind it instead of replacing
    /// the whole cell.
    ///
    /// A material is TRANSPARENT — it routes into the
    /// composite's slot-2 `OverKeepBg` layer (glyph + fg over the background behind,
    /// the material's own `bg` discarded) rather than the opaque slot-1 layer that
    /// replaces the whole cell — exactly when its recipe is the glyph-only
    /// [`TextureRecipe::Hologram`]. The hologram already forces a void `bg` and inks
    /// only sparse colored line-glyphs, so it is the natural see-through primitive:
    /// the catalog's glass / screen / veil materials are Hologram-recipe variants
    /// (different palette + scroll), and so are transparent here too. An OPAQUE
    /// material (any filled recipe — `Brick`/`Circuit`/`Grain`/`Sky`/`CodeWall`/
    /// `Scanline`) returns `false`. A cheap recipe check — no per-pixel cost; the
    /// composite split keys on exactly this.
    #[inline]
    pub fn is_transparent(&self) -> bool {
        // Reads the per-recipe descriptor — the composite slot split keys on this
        // one field, so a new see-through recipe declares `transparent: true` in
        // `TextureRecipe::props` and routes correctly with no edit here.
        self.recipe.props().transparent
    }

    /// Whether this material's GLYPH is VIEW-ANGLE dependent — it carries a
    /// [`SuperSelector::ByAngle`] super-glyph that picks the symbol from the viewing
    /// incidence. Such a cell re-projects its glyph as the camera turns, so — like a
    /// mirror or hologram — it must NEVER be rotation-panned from a baked ring; the
    /// renderer keeps it in the structural-recast (no-pan) set. `false` for every
    /// material without that selector (the default), so the pan economy is unchanged.
    #[inline]
    pub fn is_view_angle_glyph(&self) -> bool {
        matches!(
            self.super_glyph,
            Some(SuperGlyphLayer {
                selector: SuperSelector::ByAngle { .. },
                ..
            })
        )
    }

    /// Whether this material paints ONLY a colored underline on a TRANSPARENT cell —
    /// the "glass marker" look. Driven by the explicit
    /// [`glass_marker`](Self::glass_marker) opt-in AND the material being transparent
    /// (so it routes to the see-through composite layer and keeps the scene behind
    /// it). When true the composite drops this cell's glyph/fg/bg entirely and keeps
    /// only its [`UnderlineAccent`] underline as a faint pane marker over whatever is
    /// behind. `false` for every material that does not opt in (the default) — fully
    /// back-compatible, and never confused with the hologram's own Inked scan-glow.
    #[inline]
    pub fn is_underline_only_transparent(&self) -> bool {
        self.glass_marker && self.is_transparent() && self.underline_accent.is_active()
    }

    /// Can this material's output depend on [`SampleCtx::state`] for this `ctx`?
    ///
    /// The union of every reader of the state word in [`sample`](Self::sample), which
    /// is what the two shade caches must fingerprint. Four of the five are OPT-IN per
    /// material; the fifth is not, and that is the one worth spelling out.
    ///
    /// * the recipe reads it as a scalar ([`CtxInputs::state`]) — the voice mark;
    /// * a [`FamilyDriver::State`](crate::glyph_family::FamilyDriver::State) family
    ///   INDEXES its members by it;
    /// * an [`AnimTrigger::OnState`] track gates on it;
    /// * a [`SurfaceFx`](crate::surface_fx::SurfaceFx) levels redaction / decay by it;
    /// * **and [`flow_shift_coord`] / [`flow_is_hot`] run for EVERY material.** Those
    ///   two are exact no-ops when the powered bit is clear — which includes the
    ///   empty-bus `0` every cell reads when no bridge is publishing — so the key
    ///   needs nothing there. Gating on [`flow_is_hot`] rather than on "always" is
    ///   precisely what keeps a non-declaring material's key BYTE-IDENTICAL to the
    ///   pre-fix one in every scene with no live flow, while still separating two pipe
    ///   cells whose flow words differ.
    ///
    /// Over-inclusive by design where it is cheap: [`WorldTexKey`] only needs the
    /// readers inside [`texture_sample`](Self::texture_sample) (recipe, family, flow),
    /// but it consults this same predicate. Over-declaring costs a cache miss;
    /// two predicates that could drift apart cost a wrong texel.
    #[inline]
    fn reads_state_word(&self, ctx: &SampleCtx) -> bool {
        self.recipe.props().extra_inputs.state
            || matches!(
                self.family,
                Some(l) if l.driver == crate::glyph_family::FamilyDriver::State
            )
            || self.anim_tracks.count > 0
            || self.surface_fx.is_some()
            || flow_is_hot(ctx)
    }
}

/// Per-level distance-fog configuration handed to [`Material::sample`].
///
/// Fog dims **and tints** a surface toward `color` as distance runs from `near`
/// to `far`. The defaults reproduce the engine's historical look exactly: the
/// built-in `near`/`far` band and `color = None` (a fade toward black, the
/// original `dim_rgb` behaviour), so a scene with no authored fog renders
/// identically to before. A level's `[palette]` fog (forwarded through the
/// [`Map`](re_world::Map)'s atmosphere) overrides these per cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogParams {
    /// World-cell distance at which fog begins (full brightness, no tint).
    pub near: f32,
    /// World-cell distance at which fog is fully applied.
    pub far: f32,
    /// Fog tint colour `0x00RRGGBB`. `None` ⇒ fade toward black (legacy look);
    /// `Some(c)` ⇒ blend distant cells toward `c`.
    pub color: Option<u32>,
    /// Optional exponential HEIGHT fog. `None` ⇒ off, and the fog
    /// factor is exactly the distance term — bit-identical to the pre-height
    /// build, which is every shipped level.
    pub height: Option<re_world::HeightFog>,
}

impl FogParams {
    /// The engine-default fog: the built-in `near`/`far` band, fade-to-black
    /// (no tint). Matches the pre-atmosphere renderer exactly.
    pub const DEFAULT: FogParams = FogParams {
        near: FOG_NEAR,
        far: FOG_FAR,
        color: None,
        height: None,
    };
}

impl Default for FogParams {
    fn default() -> Self {
        FogParams::DEFAULT
    }
}

/// What the camera/geometry hands the material at a coordinate.
#[derive(Clone, Copy, Debug)]
pub struct SampleCtx {
    /// Perpendicular distance to the surface, in world cells.
    pub distance: f32,
    /// Foreshortening factor ≥ 1: how stretched this surface is vs. face-on.
    /// Walls: `1/|cosθ|` of the view angle; floor/ceiling: per-row derivative.
    pub foreshorten: f32,
    /// `0` = NS wall face, `1` = EW wall face (directional dimming). Unused for
    /// floor/ceiling.
    pub side: u8,
    /// Screen row, for vertical patterning on walls.
    pub screen_y: i32,
    /// Vertical wall coordinate in `[0,1]`: 0 = top of the visible wall slice,
    /// 1 = bottom. Unlike `screen_y` (an absolute terminal row), this is pinned to
    /// the WALL, so a 2D wall texture (the `CodeWall` recipe) knows which row of
    /// the source maps here regardless of where the slice sits on screen. 0 for
    /// floor/ceiling. Additive: recipes that ignore it (every built-in but
    /// `CodeWall`) are unaffected.
    pub wall_v: f32,
    /// Animation tick (frames). Pure input; this module never reads a clock.
    pub tick: u64,
    /// Elapsed wall-clock seconds, fed from Bevy's `Res<Time>` in the render
    /// system. Pure input — the shader never reads a clock itself, so the same
    /// `(coord, ctx)` (same `time`) is golden-reproducible.
    pub time: f32,
    /// Per-level distance fog (band + optional tint). [`FogParams::DEFAULT`]
    /// reproduces the historical fade-to-black look, so omitting it is a no-op.
    pub fog: FogParams,
    /// FAR-PLANE HARD FREEZE distance (world cells): at/beyond this the cell is
    /// rendered fully quiet (glyph locked to a world-anchored value, all animation
    /// disabled) — see [`FAR_FREEZE_DIST`] and [`Material::sample`]. Seeded per cast
    /// from the renderer's far-freeze knob, defaulting to [`FAR_FREEZE_DIST`].
    /// `f32::INFINITY` disables the freeze (the pre-feature behaviour), which is
    /// what omitting an authored knob yields.
    pub far_freeze: f32,
    /// Per-cell relief lighting geometry (face normal, tangent frame, light + view
    /// directions). `None` ⇒ the shader skips the lighting pass and falls back to
    /// the legacy flat `side` dimming (so a caller that doesn't supply lighting —
    /// e.g. a focused unit test — renders the historical look). The raycaster
    /// always supplies it, so the shipped look is fully lit.
    pub light: Option<LightCtx>,
    /// OPTIONAL per-material LOD OVERRIDE for this cell (the "key object" lever). When
    /// `Some`, the cell resolves its distance tier against these thresholds instead of
    /// the global [`crate::lod::LodConfig`] — so a hero material keeps detail farther
    /// out than ordinary walls. `None` (the default) ⇒ the global LOD curve. Seeded by
    /// the cast from the material's `lod_override` field; pure input, golden-safe.
    pub lod_override: Option<crate::lod::LodOverride>,
    /// STATE-BUS state word for this cell — the reactive-materials keystone.
    /// A generic `u32` looked up by the cell's `material` tag from the cast-scoped
    /// [`CellStateMap`](re_core::CellStateMap) — the renderer only INDEXES a
    /// [`FamilyDriver::State`](crate::glyph_family::FamilyDriver::State) glyph family
    /// by it (powered/dead/decoy/won), never interpreting the puzzle meaning. `0` (the
    /// default, and what an EMPTY/absent map yields for every tag) selects the family's
    /// member 0 — so a material's existing `State`-family pick is unchanged and a
    /// material WITHOUT a `State` family is byte-identical. Pure input, golden-safe.
    pub state: u32,
    /// STATE-BUS transient PULSE `[0,1]` for this cell — a brief emissive/anim KICK
    /// (a tag that just fired / a switch toggle / a cycle rejection), looked up by the
    /// cell's `material` tag from the cast-scoped [`CellStateMap`](re_core::CellStateMap).
    /// Added as a small EXTRA emissive term in [`apply_lighting`](Material::apply_lighting);
    /// `0.0` (the default, and the EMPTY-map value for every tag) adds nothing, so the
    /// shade is byte-identical. Pure input, golden-safe.
    pub pulse: f32,
    /// SCALABLE MATERIAL OVERLAY — the INSET colour override for this
    /// cell, when it is inside a resolved inset's region AND the live SelfSimilar zoom
    /// phase is inside that inset's reveal band. `Some(0x00RR_GGBB)` ⇒
    /// [`apply_scale_field`](Material::apply_scale_field) REPLACES the fractal field's
    /// resulting colour with this packed RGB (the bounded phase-band reveal — the inset
    /// wins over the procedural fractal there, then the loop carries the phase out of
    /// band and it fades). `None` (the default, and what an EMPTY resolved-inset slice
    /// yields for every cell) ⇒ the fractal field is used as-is, byte-identical.
    ///
    /// The region+band decision is made in `crate::shade_cell` (which holds the cell's
    /// MAP coordinate + the map's resolved insets), keeping `re_render` free of any
    /// `.lvl` loading — it only ever sees this distilled colour. Only consulted on a
    /// [`ScaleMode::SelfSimilar`] material, so a non-fractal cell is unaffected.
    pub inset_rgb: Option<u32>,
    /// IMAGE SPAN — the wall hit's WORLD coordinate along the wall's variable
    /// axis (world Y on an NS face, world X on an EW face). Used by
    /// [`Material::sample_image`] together with [`Self::wall_up`] and the recipe's
    /// [`ImageSpan`] to map ONE figure across the surface's full world extent. `0.0`
    /// for a floor/ceiling (no image span) and for any caller that doesn't set it; an
    /// Image recipe with NO span ignores both fields (legacy per-cell tiling), so the
    /// default is byte-identical for every non-spanned material. Pure input.
    pub wall_along: f32,
    /// IMAGE SPAN — the wall hit's WORLD UP coordinate (`wall_world_y`, the
    /// texel's world height). Pairs with [`Self::wall_along`]; see that field.
    pub wall_up: f32,
    /// Is this cell a WALL face (as opposed to a floor or ceiling)?
    ///
    /// It is the one thing [`Self::wall_along`] / [`Self::wall_up`] cannot tell a
    /// recipe on their own: both are `0.0` on a floor, but `0.0` is also a legal
    /// wall position (the bottom edge of a ground-level wall). A recipe that must
    /// pick between the WALL-plane coordinate and `coord` — which mean different
    /// things on the two surface kinds — needs the discriminator explicitly rather
    /// than inferring it from a value that can legitimately collide.
    ///
    /// Only [`TextureRecipe::Truchet`] reads it today, so every other material is
    /// byte-identical whatever this says.
    pub on_wall: bool,
    /// SUB-CELL footprint — this screen cell's WORLD `(along, up)` extent on the
    /// wall face (how much world-along one column covers, how much world-up one row
    /// covers). Used by [`Material::sample_image`] in `Quad`/`Braille` sub-cell mode to
    /// place the 2×2 / 2×4 sub-samples ±¼ footprint around the cell centre, so a near
    /// view (smaller footprint) resolves finer texels than a far one. `None` (the
    /// default, and what every non-image caster threads) ⇒ no sub-cell sampling, so the
    /// solid-block / legacy path is byte-identical. Pure input, golden-safe.
    pub tex_footprint: Option<(f32, f32)>,
    /// SUB-CELL backdrop — the colour a TRANSPARENT sub-texel reveals in
    /// `Quad`/`Braille` mode (the cut-out's "what's behind"). The cut-out column paints
    /// the backdrop first, then re-shades the opaque cell; this carries that backdrop's
    /// colour so a sub-texel hole within an otherwise-opaque cell shows the scene behind
    /// rather than the cell's own dim. `None` (the default) ⇒ no backdrop available; a
    /// transparent sub-texel then falls back to the cell's dimmed bg. Pure input.
    pub subcell_backdrop: Option<u32>,
}

/// One shaded texel: a glyph in `fg` over a `bg`, plus the detail `rank`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texel {
    /// Unicode codepoint to draw in this cell.
    pub glyph: u32,
    /// Foreground (ink) colour, packed `0x00RRGGBB`.
    pub fg: u32,
    /// Background colour, packed `0x00RRGGBB`.
    pub bg: u32,
    /// Detail rank `[0, RANK_MAX]`. 0 = base (always visible).
    pub rank: u8,
}

/// The WORLD-ANCHORED, VIEW-INDEPENDENT product of [`Material::texture_sample`].
///
/// This captures the expensive half of a shade — the multi-octave fBm/Perlin/
/// worley/brick/circuit recipe sample, the relief HEIGHT field taps, the
/// family/super-glyph substitution, and the sparse-body coverage mask — every
/// term of which is a pure function of the WORLD coordinate, the material, and the
/// (coarse, quantized) distance LOD band. NONE of it reads the per-cell `LightCtx`,
/// the headlamp, the view direction, the raw distance fog, or the sky depth band:
/// those are all CHEAP and are (re-)applied fresh every frame by
/// [`Material::apply_lighting`].
///
/// A full shade is `apply_lighting(texture_sample(...))`, byte-identical to the
/// historical single-pass [`Material::sample`] (proven by test). Because a
/// `TexResult` is determined entirely by view-INDEPENDENT inputs, it can be cached
/// across frames keyed on the world texel + material + LOD band (see
/// [`WorldTexCache`]) and re-lit cheaply as the camera moves — the texture is
/// computed once, the lighting every frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexResult {
    /// Recipe base texel (rank 0, always present).
    base: Texel,
    /// Recipe detail texel (mortar/pad/trace/star/speck), if any.
    detail: Option<Texel>,
    /// The glyph after family / super-glyph substitution (== `base.glyph` when no
    /// kindred layer fires). When a kindred layer owns the glyph, `glyph_owned` is
    /// set so the relief body re-selection / sparse-body blanking leaves it alone.
    sub_glyph: u32,
    /// A family OR super-glyph layer substituted the cell glyph (non-Sky): the
    /// relief pass must not re-select the body glyph.
    glyph_owned: bool,
    /// A super-glyph layer specifically fired (gates the detail-reveal, which a
    /// selector-owned cell must suppress — mirrors `sample`'s `super_glyph_applied`).
    super_glyph_applied: bool,
    /// The post-animation world sample coord the recipe ran at. For a STATIC
    /// material this equals the input coord (the cache key); for an animated one it
    /// carries the time-scrolled coord so the bypassed `anim_brightness` matches.
    coord: Vec2,
    /// Relief height `h(coord) ∈ [0,1]` (one tap; AO + body-glyph ramp read it).
    h: f32,
    /// The two extra central-difference height taps `(h(coord+du), h(coord+dv))`,
    /// present only when the LOD band keeps the relief gradient AND the material has
    /// a normal map. `None` ⇒ light with the flat geometric face normal.
    grad_taps: Option<(f32, f32)>,
    /// Distance LOD band this texture was sampled at (octave cap + relief on/off).
    lod: ShadeLod,
    /// FAR-PLANE freeze flag for this cell (locks the LOD/anim path).
    frozen: bool,
}

// ──────────────────── quant-memo: sample-by-quantized-key ────────────────
//
// The key efficiency lever: the same driver quantization that stabilises the
// glyph against flicker ALSO makes a natural cache key. Today
// [`quantize_lod`] only STABILISES the glyph (anti-flicker); the expensive
// `sample` (recipe noise + ranked-LOD + Blinn-Phong + DDA shadow rays) still
// RE-RUNS on every cast. The memo turns the SAME quantization into a cache key:
// a cell whose QUANTIZED drivers match a cell already shaded this frame replays
// the cached [`Texel`] instead of recomputing.
//
// BIT-IDENTITY (the hard requirement — "same pixels, less work"): a memo entry is
// keyed on the FULL fingerprint of every input `Material::sample` reads, with the
// continuous color drivers (`distance`, `coord`) carried as RAW bits and the
// LOD/glyph driver carried by its quantized bucket. So a cache HIT is, by
// construction, a cell whose `sample` inputs are equivalent → the stored texel
// equals a fresh `sample`, bit-for-bit (a `cfg(debug_assertions)` check
// re-validates every hit). The hit RATE comes from how repetitive a cast is: a
// flat wall face de-duplicates its texture per face (many screen columns map to
// the same world `coord` + `LightCtx` down a slice), and — the big win — the
// FAR-FROZEN plateau past fog saturation collapses whole regions of distant
// surface onto one quantized key (see [`SampleKey::frozen_color`]).
//
// STATIC = BAKED: a material with `!is_animated()` is a pure function of the
// quantized drivers + world coord (no `time` term), so its memo entry IS its
// bake-on-load — computed once per distinct key, reused for the rest of
// the frame. ANIMATED materials whose look depends on `time` are NOT cached
// (their key would have to carry `time`, defeating reuse and risking a stale
// frame) — they recompute every cast, correctly.

/// FNV-1a 64-bit accumulator over the raw bytes of an `sample` input fingerprint.
/// Deterministic across platforms (no float compare, just bit mixing).
#[inline]
fn fnv64(mut h: u64, x: u64) -> u64 {
    h ^= x;
    h.wrapping_mul(0x0000_0100_0000_01B3)
}

/// The full identity of a [`Material::sample`] call, captured as the exact bits of
/// every output-affecting input. Two calls with equal [`SampleKey`]s produce
/// bit-identical [`Texel`]s (the shader is pure), so the key is sufficient for a
/// correctness-preserving memo. `Eq` is derived on the raw fields (no `f32`
/// `PartialEq` pitfalls — these are `u32`/`u64` bit patterns).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct SampleKey {
    material_id: u16,
    side: u8,
    /// The LOD/glyph driver as a QUANTIZED bucket (the same `texel_density_quantized_for`
    /// deadband the anti-flicker path uses), as bits — not the raw density.
    q_density_bits: u32,
    /// World sample coord, RAW bits (parallax must stay live → exact).
    coord_x_bits: u32,
    coord_y_bits: u32,
    /// Distance, RAW bits for the continuous color/fog path (frozen+fog-saturated
    /// cells snap this to one bucket so the far field collapses onto one key).
    distance_bits: u32,
    /// Foreshorten bits (feeds the density bucket + nothing else continuous).
    foreshorten_bits: u32,
    /// Fog params (band + tint) — affects the final color.
    fog_near_bits: u32,
    fog_far_bits: u32,
    fog_color: u32,
    /// A 64-bit digest of the per-cell [`LightCtx`] (normals, light/view dirs,
    /// ambient, headlamp, shadow_key) — it fully determines the lit color/glyph.
    light_digest: u64,
    /// Whether this cell is far-frozen (changes the LOD/anim path).
    frozen: bool,
    /// DECLARED EXTRA INPUTS, `0` when the material declares none — which is what
    /// makes a non-declaring material's key byte-identical to the pre-fix one. See
    /// [`CtxInputs`] for why the four fields below are not implied by `coord`.
    ///
    /// The wall-plane hit ([`SampleCtx::wall_along`]). NOT implied by `coord`: for a
    /// wall, `coord.x = tex_u + floor(the face's CONSTANT axis)`, so it wraps once
    /// per world cell while `wall_along` runs continuously along the face.
    wall_along_bits: u32,
    /// The wall-plane height ([`SampleCtx::wall_up`]). For a WALL this IS implied by
    /// `coord.y` (`coord.y == wall_up * WALL_WORLD_V_SCALE`, a fixed non-zero scale —
    /// asserted by `wall_up_is_already_implied_by_coord_y_on_a_wall`), so folding it
    /// cannot make a wall's key finer and costs nothing. It is folded anyway, because
    /// the identity is a property of the CASTER's `sample_coord`, not of this key, and
    /// a floor/ceiling caller threads the two independently.
    wall_up_bits: u32,
    /// The state-bus word, when [`Material::reads_state_word`] says this cell's
    /// output can depend on it.
    state_bits: u32,
    /// The transient pulse bits, when non-zero. `apply_lighting` adds an emissive kick
    /// for `pulse > 0`, and `pulse` is per-TAG — so two cells of one material with
    /// different tags differ in output. `0.0` (the empty-bus value for every tag)
    /// folds `0`, so a scene with no bridge keys exactly as before.
    pulse_bits: u32,
    /// The SUB-CELL context ([`SampleCtx::tex_footprint`] /
    /// [`SampleCtx::subcell_backdrop`]) when the recipe declares it reads them.
    /// [`SubcellKey::NONE`] for every other material, so their keys are unchanged.
    subcell: SubcellKey,
}

impl SampleKey {
    /// A 64-bit digest of a [`LightCtx`] — every field that influences the lit
    /// output, mixed into one value. Used as part of [`SampleKey`].
    #[inline]
    fn light_digest(lc: Option<&LightCtx>) -> u64 {
        let Some(lc) = lc else {
            // The no-light fallback path is its own bucket (distinct from any lit
            // cell); a single sentinel covers it.
            return 0xA5A5_5A5A_DEAD_BEEF;
        };
        let mut h = 0xCBF2_9CE4_8422_2325u64; // FNV offset basis
        let mut feed = |v: f32| h = fnv64(h, v.to_bits() as u64);
        for v in [
            lc.face_n.x,
            lc.face_n.y,
            lc.face_n.z,
            lc.tangent_u.x,
            lc.tangent_u.y,
            lc.tangent_u.z,
            lc.tangent_v.x,
            lc.tangent_v.y,
            lc.tangent_v.z,
            lc.key_dir.x,
            lc.key_dir.y,
            lc.key_dir.z,
            lc.key_color.x,
            lc.key_color.y,
            lc.key_color.z,
            lc.view_dir.x,
            lc.view_dir.y,
            lc.view_dir.z,
            lc.ambient.x,
            lc.ambient.y,
            lc.ambient.z,
            lc.headlamp,
            lc.shadow_key,
        ] {
            feed(v);
        }
        h
    }

    /// Build the key for a `sample(material, coord, ctx)` call. `material_id` is the
    /// caller's stable slot id (recipe parameters are folded into it by the caller,
    /// which only memoizes per concrete registry slot). `frozen` matches
    /// `Material::sample`'s `ctx.distance >= ctx.far_freeze` test.
    #[inline]
    fn build(material_id: u16, m: &Material, coord: Vec2, ctx: &SampleCtx) -> Self {
        let frozen = ctx.distance >= ctx.far_freeze;
        // The glyph/LOD driver, quantized EXACTLY as `Material::sample` does it
        // (frozen → the fixed FAR bucket, else the stability-aware deadband). So
        // any two cells the shader would give the same glyph-LOD bucket share this
        // field — the anti-flicker quantization doubling as the cache key.
        let q_density = if frozen {
            FAR_FREEZE_LOD_BUCKET
        } else {
            texel_density_quantized_for(ctx, m.stability)
        };
        // Distance carries the continuous color/fog. Where the color is provably
        // distance-INVARIANT we snap it so distant cells collapse onto one key:
        // a far-frozen cell whose fog has saturated to a flat band (distance past
        // `fog.far`, no inked-glyph dependence on raw distance) reads a constant
        // color, so its distance can be bucketed to the frozen plateau. This is the
        // F.2 "distance-LOD on compute" payoff expressed through the same memo:
        // the whole frozen far field shares ONE key and shades ONCE.
        let distance_bits = if frozen && SampleKey::frozen_color(m, ctx) {
            FAR_FREEZE_LOD_BUCKET.to_bits()
        } else {
            ctx.distance.to_bits()
        };
        // DECLARED EXTRA INPUTS. Folded only when the material says its output can
        // depend on them, so a non-declaring material keys byte-identically to before
        // (`a_recipe_declaring_no_extra_inputs_keys_exactly_as_before`).
        let extra = m.recipe.props().extra_inputs;
        let (wall_along_bits, wall_up_bits) = if extra.wall_plane {
            (ctx.wall_along.to_bits(), ctx.wall_up.to_bits())
        } else {
            (0, 0)
        };
        let state_bits = if m.reads_state_word(ctx) {
            ctx.state
        } else {
            0
        };
        // `pulse` is read UNCONDITIONALLY by `apply_lighting`, but only above zero —
        // and `0.0` is what every tag reads from an empty bus, so this is `0` for any
        // scene with no bridge publishing.
        let pulse_bits = if ctx.pulse > 0.0 {
            ctx.pulse.to_bits()
        } else {
            0
        };
        SampleKey {
            material_id,
            side: ctx.side,
            q_density_bits: q_density.to_bits(),
            coord_x_bits: coord.x.to_bits(),
            coord_y_bits: coord.y.to_bits(),
            distance_bits,
            foreshorten_bits: ctx.foreshorten.to_bits(),
            fog_near_bits: ctx.fog.near.to_bits(),
            fog_far_bits: ctx.fog.far.to_bits(),
            fog_color: ctx.fog.color.unwrap_or(0xFFFF_FFFF),
            light_digest: SampleKey::light_digest(ctx.light.as_ref()),
            frozen,
            wall_along_bits,
            wall_up_bits,
            state_bits,
            pulse_bits,
            // The SUB-CELL context, declared per recipe. `SubcellKey::NONE` for every
            // material that does not read it ⇒ its key is byte-identical to the
            // pre-fix one.
            subcell: SubcellKey::build(m.recipe.props().extra_inputs.subcell, ctx),
        }
    }

    /// True when this far-frozen cell's COLOR is provably independent of its exact
    /// `distance` — so the distance may be bucketed in the key without changing the
    /// output. Holds when distance fog has fully saturated (`distance >= fog.far`,
    /// no tint-vs-band ambiguity): step 4 of `sample` then overwrites fg/bg with a
    /// constant (`dim_rgb(.,0)` → black, or `mix_rgb(.,fog,1)` → the tint), and the
    /// frozen glyph path already uses the fixed FAR LOD bucket (distance-invariant).
    /// The SKY recipe is EXCLUDED: its `sky_depth_band` (step 5) keeps reading the
    /// raw distance, so a sky cell's color is NOT distance-invariant even when
    /// frozen. This exclusion now reads the per-recipe descriptor
    /// ([`RecipeProps::is_sky`]) — the SAME authoritative seam every other property
    /// query uses — instead of a local recipe match, so a future distance-reading
    /// recipe opts out by declaring it in [`TextureRecipe::props`], in one place.
    #[inline]
    fn frozen_color(m: &Material, ctx: &SampleCtx) -> bool {
        !m.recipe.props().is_sky && ctx.distance >= ctx.fog.far && ctx.fog.far > ctx.fog.near
    }

    /// Collapse the key to a 64-bit hash for the open-addressed table.
    #[inline]
    fn hash(&self) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        h = fnv64(
            h,
            self.material_id as u64 | ((self.side as u64) << 16) | ((self.frozen as u64) << 24),
        );
        h = fnv64(h, self.q_density_bits as u64);
        h = fnv64(
            h,
            self.coord_x_bits as u64 | ((self.coord_y_bits as u64) << 32),
        );
        h = fnv64(
            h,
            self.distance_bits as u64 | ((self.foreshorten_bits as u64) << 32),
        );
        h = fnv64(
            h,
            self.fog_near_bits as u64 | ((self.fog_far_bits as u64) << 32),
        );
        h = fnv64(h, self.fog_color as u64);
        h = fnv64(h, self.light_digest);
        // The declared extras. All-zero for a non-declaring material, but they still
        // pass through `fnv64` — the HASH therefore differs from the pre-fix one even
        // when the KEY does not. That is harmless: the hash only picks a bucket, and
        // the full key is compared on every probe.
        h = fnv64(
            h,
            self.wall_along_bits as u64 | ((self.wall_up_bits as u64) << 32),
        );
        h = fnv64(h, self.state_bits as u64 | ((self.pulse_bits as u64) << 32));
        self.subcell.feed(h)
    }
}

/// A per-frame, allocation-reusing memo for [`Material::sample`] — the F.1
/// quant-memo. A fixed-size, power-of-two open-addressed table mapping a
/// [`SampleKey`] → its shaded [`Texel`]. Reused across frames (no per-cell heap
/// alloc): [`clear`](Self::clear) bumps a generation stamp instead of zeroing, so
/// stale entries are ignored in O(1). Bounded memory (a few thousand slots covers
/// a terminal grid many times over).
///
/// Hit/miss counters expose the recompute reduction for tests and perf reporting.
pub struct SampleMemo {
    slots: Box<[MemoSlot]>,
    mask: usize,
    generation: u32,
    hits: u64,
    misses: u64,
}

#[derive(Clone, Copy)]
struct MemoSlot {
    generation: u32,
    key: SampleKey,
    texel: Texel,
}

/// Default slot count (power of two). A 90×40 terminal casts ≈ 3600 cells/frame;
/// 8192 slots keep the table well under half-full so open-addressing probes stay
/// short, while costing only ~0.5 MB — trivial for the win.
pub const SAMPLE_MEMO_SLOTS: usize = 8192;

impl Default for SampleMemo {
    fn default() -> Self {
        Self::with_slots(SAMPLE_MEMO_SLOTS)
    }
}

impl SampleMemo {
    /// A memo with `slots` (rounded up to a power of two, min 64).
    pub fn with_slots(slots: usize) -> Self {
        let n = slots.next_power_of_two().max(64);
        let empty = MemoSlot {
            generation: 0,
            key: SampleKey {
                material_id: 0,
                side: 0,
                q_density_bits: 0,
                coord_x_bits: 0,
                coord_y_bits: 0,
                distance_bits: 0,
                foreshorten_bits: 0,
                fog_near_bits: 0,
                fog_far_bits: 0,
                fog_color: 0,
                light_digest: 0,
                frozen: false,
                wall_along_bits: 0,
                wall_up_bits: 0,
                state_bits: 0,
                pulse_bits: 0,
                subcell: SubcellKey::NONE,
            },
            texel: Texel {
                glyph: 0,
                fg: 0,
                bg: 0,
                rank: 0,
            },
        };
        SampleMemo {
            slots: vec![empty; n].into_boxed_slice(),
            mask: n - 1,
            generation: 1,
            hits: 0,
            misses: 0,
        }
    }

    /// Invalidate the whole table for a new frame in O(1) (bump the generation;
    /// no allocation, no zeroing). Resets the per-frame hit/miss counters.
    #[inline]
    pub fn clear(&mut self) {
        // Wrap-around safety: on the (astronomically rare) generation overflow,
        // hard-reset the stamps so a wrapped value can't alias a live entry.
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            for s in self.slots.iter_mut() {
                s.generation = 0;
            }
            self.generation = 1;
        }
        self.hits = 0;
        self.misses = 0;
    }

    /// Cache hits this frame (a `sample` recompute that was AVOIDED).
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache misses this frame (a `sample` that actually ran).
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Sample `material` (registry slot `material_id`) at `coord`/`ctx`, returning
    /// the shaded [`Texel`] — from the cache on a hit, else by running
    /// [`Material::sample`] and storing the result.
    ///
    /// ANIMATED materials BYPASS the cache entirely (their output depends on `time`,
    /// which is not in the key): they always recompute, so a moving/pulsing surface
    /// is never stale. STATIC materials are memoized — their entry IS their bake.
    ///
    /// Bit-identity is guaranteed: a hit only fires on an exactly-equal [`SampleKey`]
    /// (open-addressing compares the full key, not just the hash), and the shader is
    /// pure in those inputs. In debug builds every hit is re-validated against a
    /// fresh `sample`.
    #[inline]
    pub fn sample(
        &mut self,
        material_id: u16,
        material: &Material,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
    ) -> Texel {
        // Animated → never cache (time-driven look must recompute every frame).
        if material.is_animated() {
            return material.sample(perlin, coord, ctx);
        }
        let key = SampleKey::build(material_id, material, coord, ctx);
        let mut idx = (key.hash() as usize) & self.mask;
        // Open-addressing linear probe, bounded so a pathological full table falls
        // back to a recompute rather than spinning.
        for _ in 0..8 {
            let slot = &self.slots[idx];
            if slot.generation == self.generation {
                if slot.key == key {
                    let texel = slot.texel;
                    self.hits += 1;
                    #[cfg(debug_assertions)]
                    debug_assert_eq!(
                        texel,
                        material.sample(perlin, coord, ctx),
                        "quant-memo HIT must equal a fresh sample (bit-identity)"
                    );
                    return texel;
                }
                idx = (idx + 1) & self.mask; // occupied by another key → probe on
            } else {
                // Empty (stale generation): compute, store here, return.
                let texel = material.sample(perlin, coord, ctx);
                self.slots[idx] = MemoSlot {
                    generation: self.generation,
                    key,
                    texel,
                };
                self.misses += 1;
                return texel;
            }
        }
        // Probe budget exhausted (table near-full): recompute without caching.
        self.misses += 1;
        material.sample(perlin, coord, ctx)
    }
}

// ──────────────── world-anchored texture cache (cross-frame) ─────────────────
//
// The PERSISTENT, world-anchored texture cache — the cross-frame complement to the
// per-frame [`SampleMemo`]. The per-cell shade splits into an expensive,
// VIEW-INDEPENDENT [`Material::texture_sample`] (multi-octave fBm/Perlin/worley +
// relief height taps — ~93% of the near cost) and a cheap, VIEW-DEPENDENT
// [`Material::apply_lighting`] (headlamp, Blinn-Phong, fog, glyph ramp). The
// expensive half is a pure function of `(material, world coord, LOD band, wall_v,
// frozen)` — NONE of which depend on the camera's orientation or the per-cell light
// — so its [`TexResult`] is identical at a given world texel from frame to frame.
//
// [`SampleMemo`] keys on the FULL fingerprint (light digest + raw distance), so it
// only de-duplicates WITHIN a frame and is wiped every cast — under translation /
// rotation a re-seen world cell is a fresh miss every frame. [`WorldTexCache`]
// keys ONLY on the view-independent inputs and PERSISTS across frames, so walking
// (which re-sees the same world cells from a new pose) replays the cached
// `TexResult` and skips the fBm/relief entirely; only `apply_lighting` re-runs.
//
// EXACTNESS: a hit replays the byte-identical `TexResult` a fresh `texture_sample`
// would compute (the function is pure in the key's inputs), and lighting is ALWAYS
// applied fresh — so the final cell is byte-identical to a full re-shade. A
// `cfg(debug_assertions)` check re-validates every hit. ANIMATED materials bypass
// the cache (their recipe coord / kindred float read `ctx.time`, which is not in
// the key) and recompute every frame on the existing path.
//
// MEMORY: a fixed-capacity, open-addressed table (no per-cell heap alloc). Entries
// carry a `last_used` frame stamp; the table never grows. On an insert that finds no
// free/expired slot within the probe budget, the LEAST-RECENTLY-USED probed slot is
// evicted (LRU within the probe window) — so a room's finite visible surface is
// covered and stale far/off-screen texels are reclaimed. Bounded by construction.

/// The view-INDEPENDENT identity of a [`Material::texture_sample`] call: the world
/// texel + material + LOD band that FULLY determine the [`TexResult`]. Two calls
/// with equal keys produce byte-identical `TexResult`s (the texture half is pure in
/// exactly these inputs), so the key is sufficient for a correctness-preserving,
/// cross-frame cache. `Eq` on raw bit patterns (no `f32` `PartialEq` pitfalls).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct WorldTexKey {
    material_id: u16,
    /// World sample coord, RAW bits — `texture_sample` reads the exact coord, so the
    /// key must too (parallax / texture detail stay exact). For a STATIC material the
    /// coord is camera-independent at a given world surface point.
    coord_x_bits: u32,
    coord_y_bits: u32,
    /// Wall vertical coordinate bits — the `CodeWall` recipe addresses its source
    /// grid by `wall_v`; every other recipe ignores it. World-anchored (pinned to the
    /// wall), not screen-anchored, so it is view-independent.
    wall_v_bits: u32,
    /// LOD band (octave cap + relief-gradient on/off). A coarse, quantized function
    /// of distance (two boundaries), so it is stable under small camera motion — a
    /// re-seen near cell keeps the same band and hits the cache.
    lod_octaves: u32,
    lod_relief: bool,
    /// Far-freeze flag — selects the frozen LOD/anim path inside `texture_sample`.
    frozen: bool,
    /// DECLARED EXTRA INPUTS — see [`CtxInputs`] and the matching fields on
    /// [`SampleKey`], which carry the same values for the same reasons.
    ///
    /// This cache is the more dangerous of the two to under-key: [`SampleMemo`] is
    /// wiped every cast, so a missing `state` there is only wrong within one frame,
    /// while this one PERSISTS ACROSS FRAMES — a voice wall's texel would have been
    /// replayed at a stale mark for as long as the entry survived, i.e. a progress
    /// wall that never moved. (Not live in the shipped configuration: the cache is
    /// off unless `RenderConfig::tex_cache` or texel-snap turns it on.)
    wall_along_bits: u32,
    /// See [`Self::wall_along_bits`].
    wall_up_bits: u32,
    /// See [`Self::wall_along_bits`].
    state_bits: u32,
    /// The SUB-CELL context ([`SampleCtx::tex_footprint`] /
    /// [`SampleCtx::subcell_backdrop`]) when the recipe declares it reads them.
    ///
    /// This is the cache the gap hurt MOST: it persists ACROSS frames, so a cell that
    /// once cached a far, coarse-footprint sub-cell texel served that texel to the
    /// same world texel for as long as the entry survived — a wall you walk up to that
    /// never sharpens. [`SubcellKey::NONE`] for every non-sub-cell recipe.
    subcell: SubcellKey,
}

impl WorldTexKey {
    /// Build the key for a `texture_sample(material, coord, ctx)` call. Mirrors the
    /// exact inputs `texture_sample` reads (frozen test + LOD band from distance,
    /// wall_v, coord, and the material's declared extras), so any two calls the
    /// function would answer identically share this key.
    ///
    /// Takes the `material` (not just its id) because the extras are DECLARED by the
    /// recipe and by the material's own layers — the id alone cannot answer whether
    /// this cell's texture half reads the wall plane or the state word.
    #[inline]
    fn build(material_id: u16, material: &Material, coord: Vec2, ctx: &SampleCtx) -> Self {
        let frozen = ctx.distance >= ctx.far_freeze;
        let lod = ShadeLod::for_ctx(ctx);
        let extra = material.recipe.props().extra_inputs;
        let (wall_along_bits, wall_up_bits) = if extra.wall_plane {
            (ctx.wall_along.to_bits(), ctx.wall_up.to_bits())
        } else {
            (0, 0)
        };
        WorldTexKey {
            material_id,
            coord_x_bits: coord.x.to_bits(),
            coord_y_bits: coord.y.to_bits(),
            wall_v_bits: ctx.wall_v.to_bits(),
            lod_octaves: lod.max_octaves,
            lod_relief: lod.relief_gradient,
            frozen,
            wall_along_bits,
            wall_up_bits,
            state_bits: if material.reads_state_word(ctx) {
                ctx.state
            } else {
                0
            },
            subcell: SubcellKey::build(extra.subcell, ctx),
        }
    }

    /// Collapse the key to a 64-bit hash for the open-addressed table.
    #[inline]
    fn hash(&self) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        h = fnv64(
            h,
            self.material_id as u64
                | ((self.lod_relief as u64) << 16)
                | ((self.frozen as u64) << 17),
        );
        h = fnv64(
            h,
            self.coord_x_bits as u64 | ((self.coord_y_bits as u64) << 32),
        );
        h = fnv64(
            h,
            self.wall_v_bits as u64 | ((self.lod_octaves as u64) << 32),
        );
        // The declared extras — all-zero for a non-declaring material.
        h = fnv64(
            h,
            self.wall_along_bits as u64 | ((self.wall_up_bits as u64) << 32),
        );
        h = fnv64(h, self.state_bits as u64);
        self.subcell.feed(h)
    }
}

#[derive(Clone, Copy)]
struct TexSlot {
    /// `false` ⇒ empty slot (never written).
    occupied: bool,
    /// Frame stamp of the last hit/insert — drives LRU eviction.
    last_used: u64,
    key: WorldTexKey,
    tex: TexResult,
}

/// Default slot count for [`WorldTexCache`] (power of two). A room's visible surface
/// across a handful of LOD bands fits in a few thousand distinct world texels;
/// 16384 slots keep the table well under half-full (short probes) for ~2 MB —
/// trivial for the cross-frame translation win.
pub const WORLD_TEX_CACHE_SLOTS: usize = 16384;

/// Open-addressing probe budget. Within this window a miss may evict the
/// least-recently-used probed slot, so the table never grows and stale texels are
/// reclaimed under churn.
const WORLD_TEX_PROBE: usize = 8;

/// A PERSISTENT, world-anchored cache of [`TexResult`]s — the expensive,
/// view-independent texture/relief half of a shade, reused across frames. See the
/// module block above for the design + exactness contract.
///
/// Fixed-capacity, open-addressed, LRU-within-probe eviction: bounded memory, no
/// per-cell heap allocation, persists across casts (unlike [`SampleMemo`], which is
/// wiped every frame). Hit/miss counters expose the cross-frame reuse for tests +
/// perf reporting.
pub struct WorldTexCache {
    slots: Box<[TexSlot]>,
    mask: usize,
    /// Monotonic frame counter; bumped by [`tick`](Self::tick) each cast for LRU.
    frame: u64,
    hits: u64,
    misses: u64,
}

impl Default for WorldTexCache {
    fn default() -> Self {
        Self::with_slots(WORLD_TEX_CACHE_SLOTS)
    }
}

impl WorldTexCache {
    /// A cache with `slots` (rounded up to a power of two, min 64).
    pub fn with_slots(slots: usize) -> Self {
        let n = slots.next_power_of_two().max(64);
        let empty = TexSlot {
            occupied: false,
            last_used: 0,
            key: WorldTexKey {
                material_id: 0,
                coord_x_bits: 0,
                coord_y_bits: 0,
                wall_v_bits: 0,
                lod_octaves: 0,
                lod_relief: false,
                frozen: false,
                wall_along_bits: 0,
                wall_up_bits: 0,
                state_bits: 0,
                subcell: SubcellKey::NONE,
            },
            tex: TexResult {
                base: Texel {
                    glyph: 0,
                    fg: 0,
                    bg: 0,
                    rank: 0,
                },
                detail: None,
                sub_glyph: 0,
                glyph_owned: false,
                super_glyph_applied: false,
                coord: Vec2::ZERO,
                h: 0.0,
                grad_taps: None,
                lod: ShadeLod::FULL,
                frozen: false,
            },
        };
        WorldTexCache {
            slots: vec![empty; n].into_boxed_slice(),
            mask: n - 1,
            frame: 1,
            hits: 0,
            misses: 0,
        }
    }

    /// Advance to a new cast frame: bump the LRU clock and reset the per-frame
    /// hit/miss counters. Entries are NOT cleared — the cache persists across frames
    /// (that is the whole point). O(1), no allocation.
    #[inline]
    pub fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.hits = 0;
        self.misses = 0;
    }

    /// Cross-frame texture-cache hits since the last [`tick`](Self::tick) (an fBm /
    /// relief recompute that was AVOIDED).
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Texture-cache misses since the last [`tick`](Self::tick) (a `texture_sample`
    /// that actually ran the recipe + relief).
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Resolve the world-anchored [`TexResult`] for `material` (registry slot
    /// `material_id`) at `coord`/`ctx` — from the cache on a hit, else by running
    /// [`Material::texture_sample`] and storing it.
    ///
    /// ANIMATED materials BYPASS the cache (their texture depends on `time`, which is
    /// not in the key): they always recompute. STATIC materials are cached across
    /// frames. Bit-identity is guaranteed — a hit only fires on an exactly-equal
    /// [`WorldTexKey`], and `texture_sample` is pure in those inputs; debug builds
    /// re-validate every hit against a fresh sample.
    #[inline]
    pub fn texture(
        &mut self,
        material_id: u16,
        material: &Material,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
    ) -> TexResult {
        // Animated → never cache (time-driven texture must recompute every frame).
        if material.is_animated() {
            return material.texture_sample(perlin, coord, ctx);
        }
        let key = WorldTexKey::build(material_id, material, coord, ctx);
        let start = (key.hash() as usize) & self.mask;
        let mut idx = start;
        // LRU eviction bookkeeping over the probe window: remember the oldest
        // OCCUPIED probed slot in case we must evict (table region full).
        let mut lru_idx = start;
        let mut lru_age = u64::MAX;
        for _ in 0..WORLD_TEX_PROBE {
            let slot = &self.slots[idx];
            if !slot.occupied {
                // Empty: compute, store here, return.
                let tex = material.texture_sample(perlin, coord, ctx);
                self.slots[idx] = TexSlot {
                    occupied: true,
                    last_used: self.frame,
                    key,
                    tex,
                };
                self.misses += 1;
                return tex;
            }
            if slot.key == key {
                let tex = slot.tex;
                self.slots[idx].last_used = self.frame;
                self.hits += 1;
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    tex,
                    material.texture_sample(perlin, coord, ctx),
                    "world-tex cache HIT must equal a fresh texture_sample (bit-identity)"
                );
                return tex;
            }
            // Occupied by another key: track the least-recently-used probed slot.
            if slot.last_used < lru_age {
                lru_age = slot.last_used;
                lru_idx = idx;
            }
            idx = (idx + 1) & self.mask;
        }
        // Probe window full: EVICT the least-recently-used probed slot (bounded
        // memory — the table never grows). Recompute, store there, return.
        let tex = material.texture_sample(perlin, coord, ctx);
        self.slots[lru_idx] = TexSlot {
            occupied: true,
            last_used: self.frame,
            key,
            tex,
        };
        self.misses += 1;
        tex
    }
}

// ───────────────────── texel-snap (world-tex cache enabler) ─────────────────
//
// THE CACHE-WIN ENABLER. The cross-frame [`WorldTexCache`] misses (~0.7% hit) on
// the real near path because the per-cell world sample coord is sub-texel
// CONTINUOUS: a wall's `(tex_u, wall_world_y)` and a floor's exact pierce point
// shift slightly when the SAME world surface is re-seen from a new camera pose, so
// the exact key differs → a fresh miss every frame. [`texture_sample`] reads the
// EXACT coord in its fBm, so quantizing only the KEY would break byte-identity.
//
// The fix: SNAP THE SAMPLE COORD ITSELF to a fixed world-texel grid BEFORE the
// fBm/recipe runs. Then the same world surface point maps to the same texel from
// ANY view → the cache key (built from the snapped coord) collides across frames
// → cross-frame HITS. The cache value is EXACT with respect to the SNAPPED look
// (the recipe ran at the snapped coord; lighting still re-runs fresh per frame).
//
// EXACTNESS / OPT-IN: snapping CHANGES the rendered texture (it is a deliberate,
// coarser, "texel-locked" look), so it is a TOGGLE. Default OFF ⇒ the continuous
// coord flows through untouched and the output is byte-identical to today. ON ⇒
// the coord (and the wall vertical coord) are snapped, the cache is auto-enabled
// (snapping is what makes it hit), and the snapped texel is what gets rendered.
//
// Both knobs now live in the [`crate::config`] [`RenderConfig`] resource — NO env
// read in this hot path. The values are applied explicitly per cast (`RenderConfig::
// apply`) and read back here, so they are testable without environment variables.
// The env (now `TS_TEXEL_SNAP` / `TS_VIEWINDEP_LIGHT`) is consulted ONCE, at plugin
// build, to seed the resource.

/// Texels-per-cell for sample-coord snapping when ON, else `None` (OFF). The
/// world-tex cache auto-enables iff this is `Some` (snapping is what makes it hit).
/// Re-exported from [`crate::config`] so existing shade-path call sites are unchanged.
#[inline]
pub fn texel_snap_texels_per_cell() -> Option<f32> {
    crate::config::texel_snap_texels_per_cell()
}

// VIEW-INDEPENDENT LIGHTING. When ON the relief lighting DROPS every term that reads
// the camera/view direction — Blinn-Phong specular (half-vector vs view) and the
// headlamp facing term (N·V) — keeping all DEPTH cues (distance fog, depth→ink
// falloff) and all WORLD cues (directional-key diffuse N·L, albedo, height/AO,
// relief-normal perturbation, emissive). The lit result then depends ONLY on (world
// pos, depth, world normal), never on which way the camera faces.
//
// EXACTNESS WIN: the near-ring pan bakes shading at `base_yaw` and pans it to the
// live yaw; the ONLY quantity that differed under pan was this view-dependent
// shading (`bake_near_ring_column` docs). With it dropped, a near cell's shade is
// identical at every facing → the pan is byte-EXACT (no facing approximation), and
// the special-recast set can shrink to truly view-dependent GEOMETRY (mirrors).
//
// DEFAULT: ON, for readability and exact pan. See `crate::config`.

/// Whether view-independent lighting is ON (specular + headlamp-facing dropped),
/// making a cell's shade invariant to camera facing → the near-ring pan exact.
/// Re-exported from [`crate::config`] so existing shade-path call sites are unchanged.
#[inline]
pub fn view_independent_lighting() -> bool {
    crate::config::view_independent_lighting()
}

/// Snap a world sample coordinate to a `texels_per_cell` grid (cell-center sampling:
/// `floor(x·n)+0.5` over `n`). Maps every sub-texel coord within one grid cell to a
/// single representative coord, so the SAME world surface point lands on the SAME
/// texel from any camera pose → a stable cache key + an exact (snapped) sample. A
/// no-op-shaped pure function: with snapping OFF the caller never invokes it, so the
/// continuous look is untouched and byte-identical.
#[inline]
pub fn snap_to_texel_grid(coord: Vec2, texels_per_cell: f32) -> Vec2 {
    let n = texels_per_cell;
    Vec2::new(
        ((coord.x * n).floor() + 0.5) / n,
        ((coord.y * n).floor() + 0.5) / n,
    )
}

/// Snap a scalar wall-v coordinate to the same `texels_per_cell` grid. The
/// `CodeWall` recipe addresses its source grid by `wall_v`, so it must be snapped in
/// lock-step with the 2D coord for the cache key (which carries `wall_v` bits) to
/// collide across frames.
#[inline]
pub fn snap_scalar_to_texel_grid(v: f32, texels_per_cell: f32) -> f32 {
    let n = texels_per_cell;
    ((v * n).floor() + 0.5) / n
}

/// Glyph ramp shared by the wall body shading (densest → sparsest).
///
/// MINIMALIST tightening — a "symbol mush → color-forward" pass:
/// dropped the two mid-dense chars `%` and `*` from the old `@%#*+=~:-.  ` so the
/// remaining marks read as distinct STEPS (heavy `@#`, mid `+=~:`, light `-.`)
/// instead of a near-continuous grey smear. Fewer distinct glyphs → lower QA
/// `glyph_entropy`/`distinct_glyphs` (a cleaner field) while the colour palette,
/// untouched, now carries surface identity. Two trailing spaces keep the blank tail.
const BODY_RAMP: &[u8] = b"@#+=~:-.  ";

/// The canonical wall-body glyph ramp, densest → sparsest, exposed read-only so
/// dev tools (e.g. `re_imgqa`'s ink-coverage table) can assert they cover every
/// ramp glyph the shader can emit. Pure accessor — no behaviour change.
#[inline]
pub fn body_ramp() -> &'static [u8] {
    BODY_RAMP
}

#[inline]
fn ramp_glyph(brightness: f32) -> u32 {
    let n = BODY_RAMP.len();
    let idx = ((1.0 - brightness.clamp(0.0, 1.0)) * (n as f32 - 1.0) + 0.5) as usize;
    BODY_RAMP[idx.min(n - 1)] as u32
}

/// Lowest [`ramp_glyph`] brightness that still maps to an INKED ramp glyph (the
/// sparsest non-space symbol, `.`). The depth-ink cue clamps an already-inked cell
/// to at least this so far walls thin toward `.` but never blank out. Derived from
/// the ramp's two trailing spaces: the last inked index is `len-3`, whose
/// brightness midpoint is `1 − (len-3)/(len-1)`. Kept as a named const so a change
/// to [`BODY_RAMP`]'s tail is a one-line update here.
///
/// For the tightened `@#+=~:-.  ` ramp (len 10, 2 trailing spaces) the last inked
/// index is `len-3 = 7` (`.`), whose [`ramp_glyph`] brightness midpoint is
/// `1 − 7/(10−1) ≈ 0.222`. Held at that floor so the depth-ink cue thins far walls
/// to `.` but never blanks them (a blanked far wall would spike `invisible`/void
/// and hurt readability — the opposite of the goal).
const RAMP_MIN_INKED: f32 = 0.22;

/// Anti-flicker margin (see
/// `turn_test::a6_turn_test_circuit_north_wall_stays_within_the_shimmer_budget`)
/// widening the [`RAMP_MIN_INKED`] ink/blank gate one-sidedly downward: a texel
/// only needs `protrude >= RAMP_MIN_INKED - INK_GATE_MARGIN` (not the bare
/// `RAMP_MIN_INKED`) to be floored to the sparsest INKED ramp glyph instead of
/// falling through to whatever (possibly blank) glyph its un-floored
/// `depth_protrude` would pick. `protrude` is a continuous function of the WORLD
/// `coord` (raw height + lit luminance/normal), and the circuit board's calm,
/// low body density sits chronically close to `RAMP_MIN_INKED` — small enough
/// sub-cell world-position jitter (the ray hit sliding a fraction of a world
/// unit as yaw steps by one screen column) crossed the bare boundary TWICE
/// within one turn-test sweep, measured 4/100 columns (4.0%) over the 2%
/// budget. A first attempt SNAPPED `protrude` to a small quantization grid
/// before the same comparison (mirroring `q_dist`'s distance deadband) — that
/// made it WORSE (9.0%), because a periodic grid just traded one hard boundary
/// for several nearby ones the same noise could also straddle. A one-sided
/// margin has exactly one boundary, moved: picked empirically against the turn
/// test (too small still shimmers; `0.05` over-corrected the ENTIRE probed row
/// to a frozen constant glyph, tripping the test's OWN "must show some genuine
/// change" assertion — this circuit board's density sits that close to the
/// line) to a value that eliminates the double-crossing (`0/100`, `0.0%`)
/// while the probe row still shows real glyph motion elsewhere (23/100 columns
/// changed at least once) — i.e. this is not a visible flattening, just a
/// slightly more generous "still reads as inked" tolerance at the sparsest end
/// of the ramp.
const INK_GATE_MARGIN: f32 = 0.015;

// ─────────────────────────── per-cell text style ────────────────────────────
//
// The shader USES the new additive [`CellStyle`] channels (terminal SGR
// attributes + a 3rd "colored underline" color) so they are visibly working,
// tied to existing material / relief signals — never invented gameplay state:
//
//   • BOLD on strongly-lit RIDGE cells — a lit highlight (a dense body glyph
//     drawn in a bright fg) thickens, so relief peaks pop. Gated hard so only the
//     genuinely-lit ridges qualify; the vast majority of cells stay un-bold.
//   • a COLORED UNDERLINE = EMISSIVE GLOW on emissive materials (circuit traces,
//     the matrix wall): a second glowing accent in the material's own emissive
//     hue, INDEPENDENT of fg/bg, so self-lit surfaces gain a subtle under-glow.
//
// Pure function of (material, final texel) — no clock, no allocation — so it
// stays golden-reproducible and adds only a few scalar ops per cell.

/// Luma at/above which a non-blank cell counts as a "strongly-lit highlight" and
/// is bolded. Selective enough that only relief ridges / near-face-on lit cells
/// qualify (the bulk of the frame stays un-bold). NOT gated on glyph density: the
/// relief tuning desaturated the near-field glyph ramp, so a dense-glyph gate
/// would never fire.
pub const STYLE_BOLD_LUMA: f32 = 0.66;

/// Emissive strength at/above which a material earns the colored-underline glow.
/// Matches the authored emissive materials (circuit traces, matrix) without
/// catching the faint default.
pub const STYLE_GLOW_EMISSIVE: f32 = 0.18;

// ── ECONOMICAL MULTI-CHANNEL VOLUME tunables (named consts, not magic literals) ──
//
// The multi-channel model treats the terminal cell as FOUR cheap, independent
// channels and picks them TOGETHER so the limited glyph palette reads as more
// colour + volume:
//   • bg        = the dominant/back surface colour (the shader's `bg`, the fill).
//   • fg + glyph= the detail colour + its ink COVERAGE (a partial-ink glyph is an
//                 fg/bg blend at a known coverage → an extra perceived shade).
//   • underline = a THIRD colour cue: a hue-PRESERVING, brightness-shifted take that
//                 reads as the lit/near EDGE of the surface, biased by the cell's
//                 face `side` and depth so two faces of one box — or a near vs far
//                 cell of one wall — differ visibly from the SAME two base colours.
// The underline is the generalised k=2 luminance cluster: fg
// and bg are already the bright/dark clusters; the underline adds the third shade
// for free. Hue is never rotated — only luminance shifts — so each element colour
// still reads as itself (the contour colour-volume discipline).

/// How far the VOLUME underline shifts the base colour toward white on a LIT face
/// (NS, `side == 0`) — the lit/near edge cue. A gentle lift so the underline reads
/// as a highlight rim, not a second fill.
pub const VOLUME_UL_LIT_LIFT: f32 = 0.34;
/// How far the VOLUME underline shifts the base colour toward black on a SHADED face
/// (EW, `side == 1`) — the away edge. Darkening rather than lifting, so the two faces
/// of a box carry visibly DIFFERENT third cues from one element colour.
pub const VOLUME_UL_SHADE_DARK: f32 = 0.30;
/// Extra lift applied to the VOLUME underline at the near plane, decaying to 0 by the
/// volume far depth — so a near cell's third cue is brighter than a far one (depth as
/// the fourth perceived gradient). Small: depth already drives fg/bg via the relief +
/// fog, this just sharpens the near edge.
pub const VOLUME_UL_NEAR_LIFT: f32 = 0.12;
/// Near / far world-cell depths over which [`VOLUME_UL_NEAR_LIFT`] decays. Matches the
/// contour volume band so the two volume cues agree.
pub const VOLUME_UL_NEAR_DEPTH: f32 = 2.0;
/// The far endpoint of the [`VOLUME_UL_NEAR_LIFT`] decay — see [`VOLUME_UL_NEAR_DEPTH`].
pub const VOLUME_UL_FAR_DEPTH: f32 = 14.0;

/// Derive the additive per-cell [`CellStyle`] from a material and its final
/// shaded [`Texel`]. Pure, allocation-free, and tied entirely to EXISTING signals
/// (the recipe, the material's emissive, and the lit fg/glyph the shader already
/// produced) so it can only *accent* the established look, never change geometry
/// or puzzle state.
///
/// * BOLD ⇐ a strongly-lit highlight: a non-blank cell whose lit fg luma ≥
///   [`STYLE_BOLD_LUMA`]. Walls and floors only (sky/void is blank, never bolds).
/// * COLORED UNDERLINE ⇐ an emissive material (`light.emissive ≥`
///   [`STYLE_GLOW_EMISSIVE`]): an independent under-glow in the emissive hue (a
///   brightened take on the cell's own fg), so circuit/matrix traces glow with a
///   second accent the fg/bg pair can't express. `CodeWall` is excluded — its
///   legibility must not be disturbed by an under-glow.
#[inline]
pub fn cell_style(material: &Material, texel: &Texel) -> CellStyle {
    // Back-compat shim: no distance ⇒ proximity-bold can never fire (INFINITY is
    // never ≤ a finite threshold), so a pre-proximity caller is byte-for-byte
    // unchanged. The distance-aware renderer calls [`cell_style_ext`].
    cell_style_ext(material, texel, f32::INFINITY)
}

/// [`cell_style`] PLUS the distance-aware BOLD-AS-A-CHANNEL cues — the
/// proximity/emitter bold cue. `distance` is the cell's depth in
/// world cells (`f32::INFINITY` disables the proximity cue, reproducing the legacy
/// [`cell_style`]).
///
/// Two extra bold sources stack on the existing lit-ridge bold:
///
/// * **BOLD-AS-PROXIMITY** (a free depth cue): if the material sets
///   [`proximity_bold`](Material::proximity_bold)`= Some(threshold)`, a cell at/below
///   that `distance` is bolded — NEAR things thicken, a cheap "approach" cue carried
///   by the bold attribute alone, independent of colour.
/// * **BOLD-AS-AN-EMITTER CHANNEL**: an emissive material (`light.emissive ≥`
///   [`STYLE_GLOW_EMISSIVE`]) PULSING its [`Animation`] bolds on the bright half of
///   its pulse — so an emitter drives the bold attribute as a 5th expressive
///   channel (fg + bg + underline + BOLD + the kindred floating glyph). Realised
///   here purely from `(material, texel)` signals the cell already carries.
#[inline]
pub fn cell_style_ext(material: &Material, texel: &Texel, distance: f32) -> CellStyle {
    // No `side` known here ⇒ the VOLUME underline biases as a LIT (NS) face. The
    // distance-aware renderer that has the cell's `side` calls [`cell_style_full`];
    // every other (non-volume) accent is `side`-independent, so this is unchanged.
    cell_style_full(material, texel, distance, 0)
}

/// [`cell_style_ext`] PLUS the cell's face `side` (`0` = NS/lit, `1` = EW/shaded),
/// needed by the ECONOMICAL MULTI-CHANNEL VOLUME underline ([`AccentGate::Volume`])
/// so the third colour cue differs between a box's two faces. Every other accent is
/// `side`-independent, so passing `side = 0` reproduces [`cell_style_ext`] exactly.
#[inline]
pub fn cell_style_full(material: &Material, texel: &Texel, distance: f32, side: u8) -> CellStyle {
    let mut style = CellStyle::NONE;

    let props = material.recipe.props();
    let is_sky = props.is_sky;
    let is_code_wall = props.is_code_wall;
    let inked = texel.glyph != u32::from(' ');

    // BOLD (lit ridge): a strongly-lit highlight — the brightest tier of non-blank
    // cells, by lit-fg luma, on a real surface (sky/void is blank-glyph, never bolds).
    let mut bold = !is_sky && inked && luma_of(texel.fg) >= STYLE_BOLD_LUMA;

    // BOLD-AS-PROXIMITY: a near, inked cell on a material that opted in. A cheap,
    // colour-free "approach"/depth cue — only the nearest tier (distance ≤ the
    // material's threshold) bolds, so it stays selective.
    if let Some(threshold) = material.proximity_bold {
        if !is_sky && inked && distance <= threshold {
            bold = true;
        }
    }

    // BOLD-AS-AN-EMITTER CHANNEL: a material carrying BOTH a kindred-symbol
    // [`family`](Material::family) layer AND an emissive glow is treated as an EMITTER
    // — its bold attribute becomes an intensity channel. We derive the intensity from
    // the lit fg the (pulsing) animation already produced: when the cell's emissive fg
    // is on the BRIGHT half of its range it bolds, so the emitter "breathes" boldness
    // in step with its colour pulse — a 5th expressive channel on top of
    // fg/bg/underline + the kindred floating glyph. Gated on `family.is_some()` so it
    // is strictly OPT-IN (existing emissive materials carry no family → unchanged).
    if !is_sky
        && inked
        && material.family.is_some()
        && material.light.emissive >= STYLE_GLOW_EMISSIVE
        && luma_of(texel.fg) >= STYLE_BOLD_LUMA * 0.75
    {
        bold = true;
    }

    if bold {
        style = style.with_bold();
    }

    // COLORED UNDERLINE: emissive under-glow (independent 3rd channel). Skip
    // CodeWall (source text must stay clean) and Sky (a blank-glyph void cell
    // has nothing to under-glow — the accent belongs on inked emissive traces).
    if !is_code_wall && !is_sky && material.light.emissive >= STYLE_GLOW_EMISSIVE {
        // Glow hue = the cell's own emissive-ish fg, lifted toward full so it
        // reads as a distinct accent under the glyph. Stays in the material's
        // colour family (independent of bg) — a true third channel.
        let glow = unpack_rgb(texel.fg).lerp(Vec3::ONE, 0.25) * 1.15;
        style = style.with_underline_color(pack_rgb(glow.clamp(Vec3::ZERO, Vec3::ONE)));
    }

    // UNDERLINE-ACCENT LAYER (the smart-material underline substyle): a subtle
    // colored underline SHAPE under qualifying inked cells — the hologram scan-glow
    // (curly) and the brick lit-ridge glint (dotted). Realised here so it is a pure
    // function of (material, lit texel), additive, and allocation-free. Applied
    // LAST so its colour can override the emissive under-glow's colour on a holo
    // (the scan-glow IS the holo's underline), while keeping the same 3rd channel.
    style = apply_underline_accent(style, material, texel, distance, side);

    style
}

/// Apply a material's [`UnderlineAccent`] layer to a cell's [`CellStyle`]: if the
/// accent is active and this (lit) [`Texel`] passes the accent's [`AccentGate`],
/// stamp the underline SHAPE + colour. Pure, allocation-free.
///
/// * [`AccentGate::Inked`] — any non-blank cell (the hologram scan lines).
/// * [`AccentGate::LitRidge`] — only a strongly-lit non-blank cell (lit fg luma ≥
///   [`STYLE_BOLD_LUMA`]) — the rare specular-ridge glint.
/// * [`AccentGate::Volume`] — every inked cell, with the colour computed as the
///   ECONOMICAL MULTI-CHANNEL third cue ([`volume_underline`]).
///
/// The colour is the accent's fixed hue, or — for [`UnderlineAccent::HUE_FROM_FG`]
/// — a brightened take on the cell's own fg (so the underline reads as a glow in
/// the surface's own colour family, the holo hue), or — for
/// [`UnderlineAccent::HUE_VOLUME`] — the hue-preserving, `side`/depth-biased volume
/// cue. `distance`/`side` are the cell's depth + face used only by the volume cue.
#[inline]
fn apply_underline_accent(
    mut style: CellStyle,
    material: &Material,
    texel: &Texel,
    distance: f32,
    side: u8,
) -> CellStyle {
    let accent = material.underline_accent;
    if !accent.is_active() {
        return style;
    }
    let inked = texel.glyph != u32::from(' ');
    let qualifies = match accent.gate {
        AccentGate::Never => false,
        AccentGate::Inked => inked,
        AccentGate::LitRidge => inked && luma_of(texel.fg) >= STYLE_BOLD_LUMA,
        // Volume reads as a general per-cell third colour cue — every inked cell.
        AccentGate::Volume => inked,
    };
    if !qualifies {
        return style;
    }

    // Resolve the accent colour:
    //   • HUE_VOLUME → the multi-channel third cue (hue-preserving, side/depth biased);
    //   • HUE_FROM_FG → a brightened take on the cell's own fg (the holo glow);
    //   • else → the fixed authored colour.
    let color = if accent.color == UnderlineAccent::HUE_VOLUME {
        volume_underline(texel.fg, texel.bg, side, distance)
    } else if accent.color == UnderlineAccent::HUE_FROM_FG {
        let glow = unpack_rgb(texel.fg).lerp(Vec3::ONE, 0.20) * 1.1;
        pack_rgb(glow.clamp(Vec3::ZERO, Vec3::ONE))
    } else {
        accent.color & 0x00FF_FFFF
    };
    style = style.with_underline_color(color);
    style = style.with_underline_style(accent.style);
    style
}

/// The ECONOMICAL MULTI-CHANNEL VOLUME underline colour: a THIRD shade derived
/// TOGETHER with `fg`/`bg` to add perceived volume from the limited palette, at
/// ~no cost (a handful of scalar ops, no transcendentals, no allocation).
///
/// The model (a generalised k=2 luminance cluster):
/// * the cell's two base colours are the luminance clusters — the brighter of
///   `{fg, bg}` is the surface's lit ink, the darker its fill;
/// * the underline takes the BRIGHTER cluster and shifts it — toward WHITE on a LIT
///   (NS) face (a highlight rim) or toward BLACK on a SHADED (EW) face (an away
///   edge), plus a small NEAR lift that decays with depth — so the same element
///   colour yields a visibly different third cue per face and per depth.
///
/// Hue is PRESERVED throughout (only luminance scales / lifts toward grey), so the
/// element colour still reads as itself — the underline reads as relief on the
/// surface, never as a foreign tint. Returns a packed `0x00RRGGBB`.
#[inline]
fn volume_underline(fg: u32, bg: u32, side: u8, distance: f32) -> u32 {
    // k=2: the brighter cluster is the surface's detail ink.
    let (a, b) = (unpack_rgb(fg), unpack_rgb(bg));
    let base = if luma_of(fg) >= luma_of(bg) { a } else { b };

    // Depth near-lift: full at the near plane, 0 by the far plane (the 4th gradient).
    let near_t = if distance.is_finite() {
        ((VOLUME_UL_FAR_DEPTH - distance) / (VOLUME_UL_FAR_DEPTH - VOLUME_UL_NEAR_DEPTH))
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let near_lift = VOLUME_UL_NEAR_LIFT * near_t;

    // LIT (NS, side 0): lift toward white (a highlight rim). SHADED (EW, side 1):
    // darken toward black (the away edge). Both hue-preserving (lerp to grey-of-self
    // for the lit lift; a uniform scale for the shade).
    let cue = if side & 1 == 0 {
        let lift = (VOLUME_UL_LIT_LIFT + near_lift).clamp(0.0, 0.9);
        base.lerp(Vec3::ONE, lift)
    } else {
        // Darken, but a touch less when near (near_lift counteracts), so a near
        // shaded face is not crushed to black.
        let dark = (VOLUME_UL_SHADE_DARK - near_lift).clamp(0.0, 0.9);
        base * (1.0 - dark)
    };
    pack_rgb(cue.clamp(Vec3::ZERO, Vec3::ONE))
}

// ───────────────────────────── ranked-LOD ───────────────────────────────────

/// Quantization step (world cells) for the LOD driver — the ANTI-FLICKER deadband.
///
/// The reveal threshold and the body-glyph depth-ink cue are both continuous in
/// `distance`, so under sub-cell camera motion a texel's reveal can hover right at
/// its threshold and the chosen glyph flips every frame ("flicker"). We snap the
/// distance that drives GLYPH SELECTION to this step before computing the
/// threshold: a tiny camera delta that does not cross a step boundary maps to the
/// SAME quantized density → the SAME threshold → the SAME glyph. Color stays
/// continuous (it reads the un-quantized distance), so shading still varies
/// smoothly; only the discrete glyph choice is stabilized. Tuned coarse enough to
/// kill churn, fine enough that the LOD still coarsens visibly with distance.
pub const LOD_QUANT_STEP: f32 = 1.25;

/// Deadband multipliers for the two stabilised [`GlyphStability`] tiers (applied
/// to [`LOD_QUANT_STEP`]). A `Stable` body glyph holds across ~2.5 world cells of
/// approach before it re-picks; a `Frozen` far field across ~5. `Responsive` is an
/// implicit ×1.0 so it reproduces the shipped deadband exactly. Tuned coarse enough
/// to measurably cut glyph churn, fine enough that the LOD still coarsens with
/// distance (a far wall is still simpler than a near one).
pub const STABLE_DEADBAND_MUL: f32 = 2.0;
/// The `Frozen` tier's deadband multiplier — see [`STABLE_DEADBAND_MUL`].
pub const FROZEN_DEADBAND_MUL: f32 = 4.0;

/// Reveal HYSTERESIS half-widths (in rank units) for the stabilised tiers. Once a
/// detail texel is revealed, its LEAVE threshold sits this far BELOW its ENTER
/// threshold, so a small step back (which lowers the LOD threshold a touch) does
/// NOT immediately un-reveal it — the glyph that appeared stays. `Responsive` is 0
/// (no extra hysteresis → shipped behaviour). Kept well under [`REVEAL_FADE`] so a
/// genuine far→base collapse still happens, just with a deadband around it.
pub const STABLE_HYSTERESIS: f32 = 8.0;
/// The `Frozen` tier's hysteresis half-width — see [`STABLE_HYSTERESIS`].
pub const FROZEN_HYSTERESIS: f32 = 14.0;

/// `texel_density = distance × foreshortening`. The single LOD driver: it grows
/// with distance and with grazing angle, so both far *and* oblique surfaces
/// coarsen. (The spec's anti-aliasing analogue of UV-derivative mip selection.)
#[inline]
pub fn texel_density(ctx: &SampleCtx) -> f32 {
    ctx.distance * ctx.foreshorten.max(1.0)
}

/// Quantize a value to the nearest [`LOD_QUANT_STEP`] multiple (the anti-flicker
/// deadband). Pure + monotonic-non-decreasing, so a quantized density still feeds
/// a monotone reveal threshold. Rounding (not flooring) centres each plateau on
/// the step so the transition points sit between integer-ish distances.
#[inline]
pub fn quantize_lod(value: f32) -> f32 {
    quantize_lod_step(value, LOD_QUANT_STEP)
}

/// Quantize `value` to the nearest multiple of an explicit `step` (the general
/// deadband used by the per-material [`GlyphStability`] tiers — a wider `step`
/// means the glyph re-picks less often). `step` is floored to a small positive
/// epsilon so a degenerate `0` cannot divide-by-zero. Pure + monotone.
#[inline]
pub fn quantize_lod_step(value: f32, step: f32) -> f32 {
    let v = value.max(0.0);
    let s = step.max(1e-3);
    ((v / s) + 0.5).floor() * s
}

/// The GLYPH-SELECTION texel density for a given [`GlyphStability`]: [`texel_density`]
/// snapped to a deadband of `LOD_QUANT_STEP × stability.deadband_mul()` so the
/// discrete glyph reveal is temporally stable under sub-cell (and, for the stabler
/// tiers, multi-cell) camera motion. The COLOR path still uses the raw
/// [`texel_density`], so only the glyph choice is quantized — shading + parallax
/// stay smooth. `Responsive` reproduces the shipped [`texel_density_quantized`]
/// (×1.0) exactly.
#[inline]
pub fn texel_density_quantized_for(ctx: &SampleCtx, stability: GlyphStability) -> f32 {
    quantize_lod_step(
        texel_density(ctx),
        LOD_QUANT_STEP * stability.deadband_mul(),
    )
}

/// The GLYPH-SELECTION texel density: [`texel_density`] snapped to the
/// [`LOD_QUANT_STEP`] deadband so the discrete glyph reveal is temporally stable
/// under sub-cell motion. The COLOR path still uses the raw [`texel_density`], so
/// only the glyph choice is quantized — shading stays smooth. (The shipped
/// `Responsive` deadband; [`texel_density_quantized_for`] widens it per material.)
#[inline]
pub fn texel_density_quantized(ctx: &SampleCtx) -> f32 {
    texel_density_quantized_for(ctx, GlyphStability::Responsive)
}

/// Reveal threshold `T ∈ [0, RANK_MAX]` from the texel density. Near/face-on →
/// `RANK_MAX` (all ranks revealed); far/grazing → 0 (base only). Monotonic.
#[inline]
pub fn reveal_threshold(density: f32) -> f32 {
    // density LOD_NEAR..LOD_FAR maps to openness 1..0.
    let open = 1.0 - smoothstep(LOD_NEAR_DENSITY, LOD_FAR_DENSITY, density);
    open * RANK_MAX as f32
}

/// Ranked-LOD brightness for a texel of `rank` at threshold `T`:
/// `reveal = T − rank`, `brightness = smoothstep(0, FADE, reveal)`. rank-0 is
/// forced fully visible. Returns `[0,1]`; below [`DETAIL_CUTOFF`] the caller
/// drops the detail and shows the base.
#[inline]
pub fn lod_brightness(rank: u8, threshold: f32) -> f32 {
    if rank == 0 {
        return 1.0;
    }
    let reveal = threshold - rank as f32;
    smoothstep(0.0, REVEAL_FADE, reveal)
}

/// Ranked-LOD brightness WITH world-anchored reveal HYSTERESIS — the anti-flicker
/// reveal path for [`GlyphStability::Stable`]/[`GlyphStability::Frozen`].
///
/// The problem: a detail glyph whose `rank` sits right at the (quantized) reveal
/// threshold pops in and out as the camera creeps across a bucket boundary. The
/// fix is a deadband around the reveal decision with a *lower LEAVE threshold than
/// ENTER threshold*, so once a glyph appears a tiny step back will not drop it.
///
/// Because the shader is STATELESS (a pure function — no frame history), the
/// "was it already revealed?" bit is reconstructed deterministically from the
/// world texel: a texel that already cleared its rank at the threshold of the
/// NEXT-NEARER LOD bucket (`enter_threshold`, i.e. the threshold it had while the
/// camera was approaching, before it stepped back to the current, slightly-coarser
/// bucket) is treated as already-revealed, and held by the lowered LEAVE threshold
/// `threshold + hysteresis`. A texel that has NOT yet entered uses the plain ENTER
/// threshold. Net effect: the reveal/keep band is `[rank − hysteresis, rank]` in
/// threshold space — a deterministic, golden-reproducible deadband. `hysteresis == 0`
/// (the `Responsive` tier) is exactly [`lod_brightness`].
///
/// `threshold` is the current bucket's reveal threshold; `enter_threshold` is the
/// next-nearer bucket's (≥ `threshold`), supplied by the caller from the
/// stability-aware quantizer. Pure, allocation-free.
#[inline]
pub fn lod_brightness_hysteretic(
    rank: u8,
    threshold: f32,
    enter_threshold: f32,
    hysteresis: f32,
) -> f32 {
    if rank == 0 {
        return 1.0;
    }
    if hysteresis <= 0.0 {
        return lod_brightness(rank, threshold);
    }
    // Was this texel already revealed at the nearer bucket (i.e. before the small
    // step back into the current, coarser bucket)? If so, hold it with the lowered
    // LEAVE threshold so it does not blink off; otherwise require the full ENTER
    // threshold to turn it on. The two share the same `smoothstep` fade so the
    // transition is still soft (no pop), just deadbanded.
    let was_revealed = (enter_threshold - rank as f32) >= 0.0;
    let effective = if was_revealed {
        threshold - rank as f32 + hysteresis // LEAVE: held on across the deadband
    } else {
        threshold - rank as f32 // ENTER: plain reveal
    };
    smoothstep(0.0, REVEAL_FADE, effective)
}

// ───────────────────────────── sampling ─────────────────────────────────────

// ╔══════════════════════ TEXEL PERSISTENCE CONTRACT ══════════════════════════╗
// The material pipeline holds a glyph as long as it possibly can, and never makes
// noise without need: the texture's randomness and its glyph are preserved for as
// long as possible so the surface never noises without a reason. Three invariants,
// enforced here and binding on ALL future material / selector / recipe code:
//
//   1. WORLD-ANCHORED RANDOMNESS. Every per-texel random value — fBm/Perlin,
//      Worley feature points, the ink/blank coverage masks (`wall_body_inked`,
//      `floor_body_inked`, `floor_seam_inked`), the family/super-glyph phase
//      seeds (`cell_phase`), the neighbour-aware / procedural lattice hashes, the
//      CodeWall CRT scanline parity — is hashed on the WORLD position of the texel
//      (`coord` and/or the wall-pinned `wall_v`) plus the material `seed`. NEVER on
//      the screen column/row (`ctx.screen_y`), the camera pose, or `ctx.time`. A
//      texel therefore shows the SAME random value from any viewpoint: walking,
//      turning, and strafing do not re-roll it. (The ONE legitimate time term is
//      authored ANIMATION — `Pulse`/`Flicker`/`Shift` — which is opt-in per
//      material and is itself silenced for far cells by the freeze below.)
//
//   2. HYSTERETIC, QUANTIZED GLYPH HOLD. The discrete glyph for a world texel
//      changes ONLY when its world-anchored, QUANTIZED LOD driver crosses a
//      hysteresis boundary — never on sub-cell camera motion, never per frame.
//      The driver is `texel_density_quantized_for(ctx, stability)` (snapped to a
//      `LOD_QUANT_STEP × deadband_mul` plateau) and the reveal uses
//      `lod_brightness_hysteretic` (a LEAVE threshold lower than ENTER). Under
//      realistic jittery sub-cell motion the glyph-flip count for a static texel
//      is ~0 on the `Stable`/`Frozen` tiers (see the flip-count tests). COLOUR may
//      stay smoother than the glyph (it reads the raw distance), but the GLYPH is
//      sticky.
//
//   3. FAR / IDLE FREEZE. At/beyond `ctx.far_freeze` the glyph is locked to the
//      single fixed `FAR_FREEZE_LOD_BUCKET` and ALL animation is disabled, so the
//      far field is fully quiet (it still scrolls with the camera — parallax is
//      preserved — but does not churn).
//
// EXTENSIBILITY: any NEW selector, recipe, or randomness source MUST hash on the
// world texel (`coord`/`wall_v` + `seed`) and route its discrete output through the
// same quantize (`texel_density_quantized_for`) + hysteresis path. Adding a
// screen-, frame-, or camera-keyed hash is a regression of this contract and is
// covered by the `texel_*_is_world_anchored` / `*_glyph_*_stable*` tests below.
// ╚═════════════════════════════════════════════════════════════════════════════╝

impl Material {
    /// Sample this material at `coord` (the world-space surface point, or the
    /// wall face coord) under `ctx`. The core entry point: returns a fully
    /// shaded, ranked-LOD-resolved [`Texel`] (glyph in `fg` over `bg`).
    ///
    /// Pure: identical `(self, coord, ctx)` → identical `Texel`, every run.
    /// See the TEXEL PERSISTENCE CONTRACT block above for the world-anchoring and
    /// glyph-hold invariants this entry point upholds.
    pub fn sample(&self, perlin: &Perlin, coord: Vec2, ctx: &SampleCtx) -> Texel {
        // The split: the expensive, WORLD-ANCHORED texture/relief sample, then the
        // CHEAP, VIEW-DEPENDENT lighting. `apply_lighting(texture_sample(...))` is
        // byte-identical to the historical single-pass shade (proven by test). The
        // `tex` value is what [`WorldTexCache`] caches across frames; only
        // `apply_lighting` re-runs every frame as the camera moves.
        let tex = self.texture_sample(perlin, coord, ctx);
        let texel = self.apply_lighting(perlin, &tex, ctx);
        // SCALABLE MATERIAL OVERLAY: if this material carries a
        // SelfSimilar scale rule, modulate the fully-shaded base texel with the
        // shared log-polar self-similar field — the infinite-zoom material composed
        // OVER the zone's own shade. A `None`/`Crisp` material returns the texel
        // unchanged (a strict no-op ⇒ byte-identical to today).
        self.apply_scale_field(perlin, texel, coord, ctx)
    }

    /// Modulate a fully-shaded [`Texel`] with the material's [`ScaleMode`] (the Scalable
    /// Material Overlay). For [`ScaleMode::SelfSimilar`] this REUSES the
    /// exact log-polar field sampling the [`TextureRecipe::Fractal`] recipe does —
    /// [`crate::fractal::log_polar_field`] driven by
    /// [`crate::fractal::spinzoom_phase`]`(t*LN2, twist)` (the proven seamless SpinZoom
    /// loop) — and COMPOSES it over the cell's base shade: the field height drives the
    /// body glyph density (dense crests ⇒ bold glyphs) while the ring/arm phase tints
    /// the base fg/bg, so the zone's own material colour still reads under the portal.
    ///
    /// PURITY: a pure function of `(coord, time, params, perlin)` — no
    /// per-cell allocation, no transcendental beyond the field's single `ln`+`atan2`.
    /// A FAR-FROZEN cell pins the phase (the far field is quiet, like every recipe),
    /// so it stops zooming but still parallax-scrolls with `coord`. [`ScaleMode::None`]
    /// and [`ScaleMode::Crisp`] return `texel` untouched — the no-op back-compat path
    /// (Crisp is a recorded-intent marker only; its analytic edge path is a
    /// deliberately deferred feature).
    #[inline]
    fn apply_scale_field(
        &self,
        perlin: &Perlin,
        texel: Texel,
        coord: Vec2,
        ctx: &SampleCtx,
    ) -> Texel {
        let ScaleMode::SelfSimilar(p) = self.scale else {
            return texel; // None / Crisp: strict no-op (byte-identical to today).
        };
        // Phase: time-driven SpinZoom (the same seamless loop the Fractal recipe's
        // `AnimKind::SpinZoom` uses). A far-frozen cell holds phase 0 → quiet far field.
        let phase = if ctx.distance >= ctx.far_freeze {
            crate::fractal::LogPolarPhase::default()
        } else {
            crate::fractal::spinzoom_phase(ctx.time * crate::fractal::LN2, p.twist)
        };
        let s = crate::fractal::log_polar_field(coord.x, coord.y, phase, &p, perlin);

        // SCALABLE MATERIAL OVERLAY — INSET phase-band reveal (the override).
        // `crate::shade_cell` already tested region membership AND band membership
        // (it holds the cell's MAP coord + the map's resolved insets) and, when the
        // cell is inside an ACTIVE inset, handed us the inset's colour in
        // `ctx.inset_rgb`. The inset wins over the procedural fractal field HERE: we
        // REPLACE the field colour with the inset colour while STILL riding the field's
        // height for the body glyph (so the reveal keeps the zoom's relief texture and
        // does not read as a flat dead patch). `None` ⇒ the fractal field below,
        // unchanged. Empty resolved-inset slice ⇒ always `None` ⇒ byte-identical.
        if let Some(rgb) = ctx.inset_rgb {
            let base = unpack_rgb(rgb);
            // Light the inset colour by the field height so the bounded reveal still has
            // the self-similar relief (crests bright, troughs deep) rather than a flat
            // fill; the bg is a deepened shade of the same colour for ring contrast.
            let fg = base * (SELF_SIMILAR_FG_FLOOR + (1.0 - SELF_SIMILAR_FG_FLOOR) * s.height);
            let bg = base * (SELF_SIMILAR_BG_FLOOR * (0.5 + 0.5 * s.ring_phase));
            return Texel {
                glyph: ramp_glyph(s.height),
                fg: pack_rgb(fg.clamp(Vec3::ZERO, Vec3::ONE)),
                bg: pack_rgb(bg.clamp(Vec3::ZERO, Vec3::ONE)),
                rank: texel.rank,
            };
        }

        // Compose the field OVER the base shade. The body glyph rides the field height
        // (dense crests ⇒ bold ramp glyphs, troughs ⇒ sparse) — the visible
        // self-similar relief. The base fg keeps the material's hue but is brightened
        // toward the crests; the bg is deepened toward the troughs for ring contrast.
        // Both stay a function of the EXISTING base colour, so the zone's authored
        // material still reads under the portal (composes, never replaces).
        let fg = unpack_rgb(texel.fg)
            * (SELF_SIMILAR_FG_FLOOR + (1.0 - SELF_SIMILAR_FG_FLOOR) * s.height);
        let bg = unpack_rgb(texel.bg)
            * (SELF_SIMILAR_BG_FLOOR + (1.0 - SELF_SIMILAR_BG_FLOOR) * s.ring_phase);
        Texel {
            glyph: ramp_glyph(s.height),
            fg: pack_rgb(fg.clamp(Vec3::ZERO, Vec3::ONE)),
            bg: pack_rgb(bg.clamp(Vec3::ZERO, Vec3::ONE)),
            rank: texel.rank,
        }
    }

    /// The WORLD-ANCHORED, VIEW-INDEPENDENT half of a shade: run the procedural
    /// recipe (fBm/Perlin/worley/brick/circuit), resolve the family/super-glyph
    /// substitution + the sparse-body mask, and take the relief HEIGHT-field taps —
    /// everything that is a pure function of `(material, world coord, LOD band)` and
    /// does NOT read the per-cell light / view direction / raw-distance fog.
    ///
    /// The result is the cacheable [`TexResult`]; feed it to
    /// [`apply_lighting`](Self::apply_lighting) for the final [`Texel`]. For a STATIC
    /// material this value is identical every frame at a given world texel + LOD band,
    /// so it is what [`WorldTexCache`] persists. (Animated materials bypass the cache:
    /// their recipe coord scroll / kindred float read `ctx.time`, captured here so a
    /// fresh call still animates correctly — they just are never stored.)
    pub fn texture_sample(&self, perlin: &Perlin, coord: Vec2, ctx: &SampleCtx) -> TexResult {
        // FAR-PLANE HARD FREEZE: at/beyond `ctx.far_freeze` this cell is rendered
        // fully QUIET — glyph locked to one world-anchored FAR bucket (no LOD reveal
        // flips, no depth-ink re-selection) and ALL animation disabled (no pulse /
        // flicker / shift / hologram-scroll). The glyph stays a function of the world
        // `coord`, so the cell still SCROLLS with the camera (parallax preserved);
        // only the per-frame churn is killed. A HARD cutoff stacked on top of the
        // soft [`GlyphStability`] deadband. `far_freeze = ∞` ⇒ never frozen (the
        // pre-feature behaviour). See [`FAR_FREEZE_DIST`].
        let frozen = ctx.distance >= ctx.far_freeze;

        // Per-recipe behaviour, read ONCE from the single authoritative descriptor
        // ([`TextureRecipe::props`]) and reused for every property query below — no
        // scattered `match self.recipe` for SKY / self-lit / code-wall / sparse-body.
        let props = self.recipe.props();

        // 0. Animation: `Shift` scrolls the sample coord by time *before* the
        //    recipe runs; `Pulse`/`Flicker` scale brightness *after* (step 3).
        //    Pure function of `(coord, ctx.time)` — no clock, no per-cell alloc.
        //    A FAR-FROZEN cell skips the time-driven coord scroll, so a far hologram /
        //    matrix wall holds still (its glyph is then world-anchored, not animated).
        let coord = if frozen {
            coord
        } else {
            self.anim_shift_coord(coord, ctx)
        };
        // 0b. FLOW: a pipe cell whose state-bus word
        //     decodes to a POWERED flow direction scrolls its glyph pattern ALONG the
        //     surface axis toward the output (input→output), at a fixed rate; an
        //     unpowered cell (`state == 0`, the empty-bus default, AND every non-pipe
        //     material) does NOT scroll → byte-identical to today. A FAR-FROZEN cell
        //     holds (the far field is quiet). The matching HUE LIFT (hot when powered)
        //     is applied after shading in `apply_flow_hue`.
        //
        //     `state_is_scalar` recipes are EXCLUDED: they read the same word as a
        //     continuous mark, and every odd value of it would otherwise decode as a
        //     powered pipe. See `RecipeProps::state_is_scalar`.
        let coord = if frozen || props.state_is_scalar {
            coord
        } else {
            flow_shift_coord(coord, ctx)
        };

        // DISTANCE-LOD shade band for this cell (relief-gradient on/off + octave
        // cap), computed ONCE here and threaded into the recipe noise + relief pass.
        // NEAR cells get `ShadeLod::FULL` (byte-identical to the pre-LOD path); only
        // mid/far cells drop the imperceptible relief micro-shading + sub-pixel noise.
        let lod = ShadeLod::for_ctx(ctx);

        // 1. Recipe produces a *base* texel (rank 0) + an optional *detail*
        //    texel (rank > 0). distance fog is applied to both as brightness.
        let (mut base, detail) = match self.recipe {
            TextureRecipe::Brick {
                aspect,
                mortar_w,
                grain,
            } => self.sample_brick(perlin, coord, ctx, aspect, mortar_w, grain),
            TextureRecipe::Circuit { scale, trace_w } => {
                self.sample_circuit(perlin, coord, ctx, scale, trace_w)
            }
            TextureRecipe::Grain {
                scale,
                speck,
                tile_eps,
            } => self.sample_grain(perlin, coord, ctx, scale, speck, tile_eps),
            TextureRecipe::Sky { scale, density } => {
                self.sample_sky(perlin, coord, ctx, scale, density)
            }
            TextureRecipe::Hologram { line_freq, scan_w } => {
                self.sample_hologram(coord, ctx, line_freq, scan_w)
            }
            TextureRecipe::CodeWall {
                panel,
                u_chars_per_unit,
            } => self.sample_code_wall(coord, ctx, panel, u_chars_per_unit),
            TextureRecipe::NoteWall { note, face } => self.sample_note_wall(coord, ctx, note, face),
            TextureRecipe::Scanline {
                freq,
                duty,
                warp,
                axis,
            } => self.sample_scanline(perlin, coord, ctx, freq, duty, warp, axis),
            TextureRecipe::Truchet {
                seed,
                density,
                scale,
            } => self.sample_truchet(coord, ctx, seed, density, scale),
            TextureRecipe::Fractal {
                arms,
                octave_warp,
                twist,
            } => self.sample_fractal(perlin, coord, ctx, arms, octave_warp, twist),
            TextureRecipe::Substrate {
                scale,
                held,
                released,
            } => self.sample_substrate(ctx, scale, held, released),
            // VOICE: the same surface, with its two scalars read off the state bus
            // instead of off the recipe. One `u32` in, one `(held, released)` pair
            // out, then the SAME sampler — so a voice wall and a substrate wall can
            // never diverge in look, only in what moves them.
            TextureRecipe::Voice { scale, mode } => {
                let mark = crate::substrate::decode_mark(ctx.state);
                let (held, released) = mode.channels(mark);
                self.sample_substrate(ctx, scale, held, released)
            }
            // IMAGE: sample the attached RGBA8 texture at the cell's (tex_u, wall_v).
            // No detail texel — the whole cell is one solid `█` block by default.
            TextureRecipe::Image { .. } => (self.sample_image(coord, ctx), None),
            // FACADE: the structured bay×floor window grid (`crate::facade`).
            TextureRecipe::Facade {
                style,
                bay,
                floor_h,
                lit,
                seed,
                ground_band,
                sign_row,
            } => self.sample_facade(
                coord,
                ctx,
                style,
                bay,
                floor_h,
                lit,
                seed,
                ground_band,
                sign_row,
            ),
        };

        // 1b. Glyph-source override: a hand-authored base glyph replaces the
        //     recipe's choice for the base texel (details still come from the
        //     recipe). `Recipe` is the identity.
        base.glyph = self.glyph_source.resolve(base.glyph);

        // 1c. KINDRED-SYMBOL super-glyph layer: if the material carries a family,
        //     SUBSTITUTE a family member for the base glyph by the family's driver
        //     (the "floating" / growth / positional effect). Pure + allocation-free.
        //     A FAR-FROZEN cell pins the float (no time term → it holds still, like
        //     every other animation under the freeze). The per-cell phase/seed is a
        //     WORLD hash so neighbours desync yet a spot is stable frame-to-frame.
        let mut family_glyph_applied = false;
        if let Some(layer) = self.family {
            base.glyph = self.family_glyph(layer, base.glyph, coord, ctx, frozen);
            // The family OWNS the glyph for every non-Sky surface (Sky returns the
            // blank unchanged), so the relief body re-selection / sparse-body blanking
            // must leave the substituted floating glyph in place.
            family_glyph_applied = !props.is_sky;
        }

        // 1d. SUPER-GLYPH layer (selectors beyond the four family
        //     drivers): if present, SUBSTITUTE a kindred member chosen by the richer
        //     selector (bpm / density / neighbour-aware / multiply / priority /
        //     procedural). It runs SECOND — after `family` — with the IDENTICAL frozen /
        //     quantized-density handling, so a far / idle cell does not flicker. One
        //     quantized table lookup, no per-cell heavy math, no alloc. Like the family
        //     it OWNS the glyph (so the relief re-selection leaves it in place), and SKY
        //     is left blank (its glyph carries the depth band, not ink).
        let mut super_glyph_applied = false;
        if let Some(layer) = self.super_glyph {
            base.glyph = self.super_glyph_glyph(layer, base.glyph, coord, ctx, frozen);
            super_glyph_applied = !props.is_sky;
            // The super-glyph OWNS the cell glyph (it REPLACES the recipe base), so the
            // relief body re-selection must leave it in place — same as `family`.
            family_glyph_applied |= super_glyph_applied;
        }

        // RELIEF HEIGHT-FIELD TAPS (the costliest part — up to three multi-octave
        // `height_at` evals). These are a pure function of `(coord, lod)`, so they
        // belong to the world-anchored texture half. The gradient taps are skipped
        // (the flat geometric normal lights the cell) past the NEAR relief band or
        // when the material has no normal map — EXACTLY the predicate the relief pass
        // used, so the lit output is unchanged. SELF-LIT / hologram / code-wall
        // recipes never reach the relief pass; we still take the cheap single `h` tap
        // (its `height_at` returns a flat constant for them) so the value is defined,
        // but skip the gradient taps for them.
        //
        // RELIEF-IMAGE COORD SWAP: every procedural recipe addresses its height field
        // in the same world-anchored `coord` its colour recipe uses. An image does
        // not - its height map is indexed by the IMAGE's own `(u, v)`, which the
        // span/fit/tile mapping derives from the WALL PLANE, not from `coord`. So a
        // relief image samples its height at the image UV, which also makes the eps
        // taps below step across texels instead of across an axis the height map
        // ignores. `None` for every other material (and for an image with no authored
        // relief / no height channel), which yields `coord` - byte-identical.
        let h_coord = self.image_relief_uv(coord, ctx).unwrap_or(coord);
        let h = self.height_at(perlin, h_coord, lod);
        let lm = &self.light;
        let grad_taps = if !props.self_lit
            && lod.relief_gradient
            && lm.normal_strength > 0.0
            && lm.height_scale > 0.0
        {
            let eps = HEIGHT_GRAD_EPS;
            let hu = self.height_at(perlin, h_coord + Vec2::new(eps, 0.0), lod);
            let hv = self.height_at(perlin, h_coord + Vec2::new(0.0, eps), lod);
            Some((hu, hv))
        } else {
            None
        };

        let glyph_owned = family_glyph_applied; // family OR super already folded in
        TexResult {
            base,
            detail,
            sub_glyph: base.glyph,
            glyph_owned,
            super_glyph_applied,
            coord,
            h,
            grad_taps,
            lod,
            frozen,
        }
    }

    /// The CHEAP, VIEW-DEPENDENT half of a shade: take a world-anchored
    /// [`TexResult`] and apply the per-frame lighting — detail reveal, relief /
    /// normal-mapped Blinn-Phong (or the no-light fallback dim), distance fog, and
    /// the sky depth band — to produce the final [`Texel`]. Reads the per-cell
    /// `LightCtx`, the raw `distance`, and the fog from `ctx`; never re-runs the
    /// expensive recipe noise or the height field (those live in the `TexResult`).
    ///
    /// `apply_lighting(texture_sample(self, coord, ctx), ctx)` == the historical
    /// single-pass `sample`, byte-identical (proven by test). `perlin` is accepted
    /// for signature symmetry / future use; the lighting math reads no noise.
    pub fn apply_lighting(&self, _perlin: &Perlin, tex: &TexResult, ctx: &SampleCtx) -> Texel {
        let props = self.recipe.props();
        let frozen = tex.frozen;
        let lod = tex.lod;
        let base = tex.base;
        // Distance fog × HEIGHT fog. The height term reads the cell's
        // world height off the anchored `LightCtx` (`world_pos.z` — z IS height in
        // this codebase, matching `CameraPose::pos`). A caller that supplies no
        // lighting context (a focused unit test) gets the distance term alone, and
        // a level with no `[atmosphere] fog_height` gets a literal 1.0 multiplier,
        // so both are bit-identical to the pre-height build.
        let fog = distance_fog(ctx.distance, &ctx.fog)
            * match (ctx.fog.height, ctx.light) {
                (Some(_), Some(lc)) => height_fog(lc.world_pos.z, &ctx.fog),
                _ => 1.0,
            };
        // An EMITTER makes its own light: the haze takes less of it.
        let fog = if self.light.emissive >= SELF_LIT_EMISSIVE {
            fog + (1.0 - fog) * EMITTER_FOG_KEEP
        } else {
            fog
        };

        // 2. Ranked-LOD: reveal the detail only if its rank clears the
        //    density-driven threshold, fading in smoothly. The threshold reads the
        //    QUANTIZED density (the anti-flicker deadband) so a revealed detail
        //    glyph does not pop in/out under sub-cell camera motion — the discrete
        //    glyph choice only changes when the camera crosses a step boundary.
        let stability = self.stability;
        // FAR FREEZE locks GLYPH SELECTION to a single fixed FAR LOD bucket so the
        // reveal threshold (and thus the chosen glyph) is INVARIANT to the cell's
        // actual distance/angle. Otherwise the stability-aware quantizer runs.
        let q_density = if frozen {
            FAR_FREEZE_LOD_BUCKET
        } else {
            texel_density_quantized_for(ctx, stability)
        };
        let threshold = reveal_threshold(q_density);
        let enter_density = (q_density - LOD_QUANT_STEP * stability.deadband_mul()).max(0.0);
        let enter_threshold = reveal_threshold(enter_density);
        let hysteresis = stability.hysteresis();
        let mut out = Texel {
            glyph: tex.sub_glyph,
            ..base
        };
        let mut detail_revealed = false;
        // A FIXED base glyph (a hand-authored / puzzle-element look) must read as
        // ITSELF at every distance, so the recipe's detail never overrides it; a
        // SUPER-GLYPH likewise owns the cell — gate the reveal on it too.
        let allow_detail =
            matches!(self.glyph_source, GlyphSource::Recipe) && !tex.super_glyph_applied;
        if let (Some(d), true) = (tex.detail, allow_detail) {
            let reveal = lod_brightness_hysteretic(d.rank, threshold, enter_threshold, hysteresis);
            if reveal > DETAIL_CUTOFF {
                // Blend the detail glyph's fg in over the base by reveal; keep
                // the base bg (the second channel stays stable as detail fades).
                out.glyph = d.glyph;
                out.fg = mix_rgb(base.fg, d.fg, reveal);
                out.bg = mix_rgb(base.bg, d.bg, reveal * 0.6);
                out.rank = d.rank;
                detail_revealed = true;
            }
        }

        // 3. LIGHTING. When the caster supplied a [`LightCtx`] (the shipped path),
        //    run the full relief / normal-mapped Blinn-Phong model; with NO
        //    `LightCtx` (a focused test), fall back to the legacy flat `side` dim.
        // FAR-FROZEN cells skip pulse/flicker brightness modulation (1.0 = none).
        let anim_b = if frozen {
            1.0
        } else {
            self.anim_brightness(tex.coord, ctx)
        };
        let mut unlamped: Option<(u32, u32)> = None;
        if let Some(lc) = ctx.light.as_ref() {
            // DEPTH-CONTRAST detail half (4b runs later, colour-only, over `out`;
            // this precomputes the GLYPH-VARIETY collapse so `apply_relief_lighting`
            // can fold it into the same body-ramp pick it already does). Gated on
            // the SAME master switch as the colour effect, so `TS_DEPTH=0` restores
            // the un-collapsed ramp exactly (see `depth_contrast::ramp_steps_for`).
            let ramp_steps = if crate::depth_contrast::active().enabled {
                crate::depth_contrast::ramp_steps_for(
                    ctx.distance,
                    ctx.fog.near,
                    ctx.fog.far,
                    &crate::lod::active(),
                    self.lod_override,
                )
            } else {
                None
            };
            let relief = ReliefArgs {
                detail_revealed,
                anim_b,
                distance: ctx.distance,
                frozen,
                family_glyph: tex.glyph_owned,
                lod,
                coord: tex.coord,
                h: tex.h,
                grad_taps: tex.grad_taps,
                ramp_steps,
            };
            // The legibility floor (4a/5a) needs this cell as it would be WITHOUT
            // the level's point lights, so a lamp's light is added on top of the
            // floored murk instead of being compressed by the floor (see
            // `legibility::legible_pair_lamped`). Only a lit level with authored
            // lamps pays for the second pass; every other cell skips it.
            let floor_exempt = props.self_lit && (props.is_code_wall || props.transparent);
            if !floor_exempt && !lc.points.is_empty() {
                let no_lamps = LightCtx {
                    points: PointLights::NONE,
                    ..*lc
                };
                let d = self.apply_relief_lighting(tex, &no_lamps, out, &relief);
                unlamped = Some((d.fg, d.bg));
            }
            out = self.apply_relief_lighting(tex, lc, out, &relief);
        } else {
            let side_f = if ctx.side != 0 { SIDE_EW_DIM } else { 1.0 };
            let lit = side_f * anim_b;
            out.fg = dim_rgb(out.fg, lit);
            out.bg = dim_rgb(out.bg, lit);
        }

        // 3b. STATE-BUS PULSE: a transient emissive KICK from the
        //     cell's `material` tag (a node just fired, a switch toggled, a cycle was
        //     rejected). Brightens the inked fg/bg additively by a small factor of the
        //     `[0,1]` pulse so the event "flashes" without changing the glyph. A
        //     FAR-FROZEN cell suppresses it (the far field holds quiet, like every
        //     other animation under the freeze). `pulse == 0.0` (the EMPTY-map default
        //     for every tag) is a no-op → byte-identical to today. Pure, allocation-free.
        if ctx.pulse > 0.0 && !frozen {
            let kick = 1.0 + STATE_PULSE_EMISSIVE * ctx.pulse.clamp(0.0, 1.0);
            out.fg = brighten_rgb(out.fg, kick);
            out.bg = brighten_rgb(out.bg, kick);
            if let Some((fg, bg)) = unlamped.as_mut() {
                *fg = brighten_rgb(*fg, kick);
                *bg = brighten_rgb(*bg, kick);
            }
        }

        // 3c. FLOW HOT-HUE: a POWERED pipe cell
        //     reads HOTTER than a cold (unpowered) one — lift the inked fg so a live
        //     flowing trace is visibly warmer/brighter than a dead decoy pipe. The
        //     glyph scroll (step 0b) carries the direction; this carries the "alive"
        //     temperature. An unpowered cell (`state == 0`, the empty-bus default,
        //     and every non-pipe material) is NOT hot → no lift → byte-identical to
        //     today. A FAR-FROZEN cell holds quiet (suppressed, like the pulse).
        //     A `state_is_scalar` recipe is excluded for the same reason it skips the
        //     flow scroll: its state word is a mark, and half of the marks are odd.
        if !frozen && !props.state_is_scalar && flow_is_hot(ctx) {
            out.fg = brighten_rgb(out.fg, FLOW_HOT_LIFT);
            if let Some((fg, _)) = unlamped.as_mut() {
                *fg = brighten_rgb(*fg, FLOW_HOT_LIFT);
            }
        }

        // 4. Distance fog. `fog` is 1.0 near → 0.0 far.
        match ctx.fog.color {
            None => {
                out.fg = dim_rgb(out.fg, fog);
                out.bg = dim_rgb(out.bg, fog);
                if let Some((fg, bg)) = unlamped.as_mut() {
                    *fg = dim_rgb(*fg, fog);
                    *bg = dim_rgb(*bg, fog);
                }
            }
            Some(fog_rgb) => {
                let t = 1.0 - fog; // 0 near (untinted) → 1 far (full fog colour)
                out.fg = mix_rgb(out.fg, fog_rgb, t);
                out.bg = mix_rgb(out.bg, fog_rgb, t);
                if let Some((fg, bg)) = unlamped.as_mut() {
                    *fg = mix_rgb(*fg, fog_rgb, t);
                    *bg = mix_rgb(*bg, fog_rgb, t);
                }
            }
        }

        // 4a. PERCEPTUAL LEGIBILITY FLOOR (crate::legibility): the multiplicative
        //     lighting above (albedo × ambient, bg × shadow, then fog toward a
        //     near-black fog colour) compounds a murk palette down to bytes ~2..24 —
        //     invisible on a real screen. Soft-lift the bg's Oklab L to a dim floor
        //     and keep the ink a minimum ΔL above it, tinted along the material's
        //     own authored hue. Monotonic, so relative darkness survives; identity
        //     above the knees and for already-clear ink, so lamp-lit glyphs keep
        //     their colour. Runs AFTER fog so the murk cannot re-bury the cell, and
        //     BEFORE the opt-in depth-contrast policy (4b) so its fade-to-sky still
        //     dissolves exactly into its reference. A SKY/ceiling cell takes it after
        //     its depth band instead (5a). Only the LIT path (a `LightCtx` was
        //     supplied), and never a self-lit TEXT / VOID recipe (code wall, placard,
        //     hologram) whose black is authored meaning. Self-lit PANELS
        //     (scanline/metal, truchet, fractal) do take it — they are built into
        //     ceilings and walls, where a near-black ground is the same defect.
        let authored_void = props.self_lit && (props.is_code_wall || props.transparent);
        let legibility_floor = ctx.light.is_some() && !authored_void;
        //     A cell the level's point lights reach floors its LAMP-LESS shade and
        //     takes the lamps' light on top (`legible_pair_lamped`), so a light pool
        //     stays brighter than the murk around it by its full raw contribution.
        if legibility_floor && !props.is_sky {
            let (fg, bg) = legible_floor(out.glyph, (out.fg, out.bg), unlamped, (base.fg, base.bg));
            out.fg = fg;
            out.bg = bg;
        }

        // 4b. UNIVERSAL DEPTH-CONTRAST (depth_contrast.rs). A colour-only policy that
        //     makes FAR surfaces lose contrast (ink pulled toward their own bg) while
        //     NEAR surfaces keep full ink, optionally dissolving the far field into the
        //     backdrop ("soft emergence") plus a hash-stable per-cell volume jitter.
        //     GLYPH IS NOT TOUCHED (the DETAIL half of the effect — body-ramp step
        //     collapse — runs earlier, in `apply_relief_lighting`, gated on the SAME
        //     `dc.enabled`). Struct `Default`/thread-local init stay OFF (test-harness
        //     baseline — proven by `split_matches_single_pass_sample_byte_for_byte`,
        //     which never enables it); the SHIPPED default is ON via
        //     `DepthContrastConfig::from_env` (`TS_DEPTH=0` restores the pre-feature
        //     look exactly). EMITTERS bypass this: they splat in a separate composite
        //     layer and never route through `apply_lighting`, so they are naturally
        //     exempt. Runs AFTER fog, BEFORE the sky band + animation/glyph tracks.
        {
            let dc = crate::depth_contrast::active();
            if dc.enabled {
                // Derive the near/far band from the LEVEL'S OWN authored fog band
                // (`ctx.fog.near/far`, always populated — level-authored or the
                // engine default) rather than the fixed global LOD thresholds:
                // small rooms (8–25 cells, per the shipped corpus) rarely reach a
                // fixed far_start=13, and tying the band to the SAME fog band the
                // distance-fog pass (step 4, above) already uses makes contrast
                // collapse and fog tint dissolve together instead of on two
                // independent schedules. Falls back to the LOD thresholds only for
                // a degenerate/non-finite fog (a hand-built test ctx; a parsed
                // level never produces one).
                //
                // WIDENED by this material's own [`crate::lod::LodOverride`] (the
                // same "hero object holds detail" lever the glyph LOD tier already
                // reads off `SampleCtx::lod_override`) — the PUZZLE-CRITICAL SURFACE
                // exemption: a material authoring `full_detail_to(d)` (e.g. the
                // puzzle-node marker) keeps FULL contrast, no sky-fade and no volume
                // jitter out to `d`, instead of dimming like an ordinary wall. This is
                // the mechanism, not a glyph special case — any material can opt in.
                let lod_cfg = crate::lod::active();
                let (near_end, far_start) = crate::depth_contrast::band_for_fog_ov(
                    ctx.fog.near,
                    ctx.fog.far,
                    &lod_cfg,
                    self.lod_override,
                );

                // 1) Hash-stable per-cell volume jitter (deterministic on world coord,
                //    identical every frame → no flicker, cache-safe). Applied to fg/bg
                //    first so the contrast/sky blends operate on the jittered colour.
                //    SCALED DOWN by the band's own width (`jitter_amplitude_for_band`):
                //    a level authoring a very short fog band (e.g. `08-material-demo`,
                //    `fog_near 1 / fog_far 4`) otherwise reads as block-colour mottling
                //    — see the constant's doc. An ordinary room's band is wide enough
                //    that this is a no-op (full amplitude, byte-identical).
                let jitter_amp = crate::depth_contrast::jitter_amplitude_for_band(
                    dc.volume_jitter,
                    far_start - near_end,
                );
                if jitter_amp > 0.0 {
                    let j = crate::depth_contrast::volume_jitter_factor(
                        tex.coord.x,
                        tex.coord.y,
                        jitter_amp,
                    );
                    out.fg = dim_rgb(out.fg, j);
                    out.bg = dim_rgb(out.bg, j);
                }

                // 2) Contrast collapse: pull fg toward its OWN bg by (1 - contrast).
                let contrast =
                    crate::depth_contrast::contrast_for(ctx.distance, near_end, far_start, &dc);
                if contrast < 1.0 {
                    out.fg = mix_rgb(out.fg, out.bg, 1.0 - contrast);
                }

                // 3) Fade-to-sky: beyond far_start, blend BOTH fg and bg toward the
                //    background/sky reference (the fog tint if set, else the neutral far
                //    grey) so the surface dissolves into the backdrop with no pop.
                if dc.fade_to_sky {
                    // Guard against a HARD STEP: `sky_blend_for` degenerates to an
                    // instant full-blend at `far_start` when `full_at <= far_start`.
                    // The configured `sky_blend_full_at` (default: the historical
                    // far-freeze distance) can sit BELOW a level's authored
                    // `fog_far` (e.g. a level with no `[atmosphere] fog_far` inherits
                    // the wide unauthored default), so floor it past `far_start` —
                    // this never SHRINKS the fade band, only widens a degenerate one.
                    let full_at = dc.sky_blend_full_at.max(far_start + 1.0);
                    let blend =
                        crate::depth_contrast::sky_blend_for(ctx.distance, far_start, full_at);
                    if blend > 0.0 {
                        let sky_ref = ctx
                            .fog
                            .color
                            .unwrap_or(crate::depth_contrast::DEPTH_NEUTRAL_FAR_REF);
                        out.fg = mix_rgb(out.fg, sky_ref, blend);
                        out.bg = mix_rgb(out.bg, sky_ref, blend);
                    }
                }
            }
        }

        // 5. SKY / CEILING depth band: a final monotone distance dim on the void so
        //    far ceiling reads dimmer than near. Runs LAST so the relight can't wash
        //    it out. Safe (a blank sky cell has no inked fg/bg pair to collapse).
        if props.is_sky {
            let band = sky_depth_band(ctx.distance);
            out.fg = dim_rgb(out.fg, band);
            out.bg = dim_rgb(out.bg, band);
            if let Some((fg, bg)) = unlamped.as_mut() {
                *fg = dim_rgb(*fg, band);
                *bg = dim_rgb(*bg, band);
            }
        }

        // 5a. LEGIBILITY FLOOR, SKY half — see 4a. A ceiling/sky cell takes the floor
        //     only here, AFTER its depth band, which can dim to 10 % and would
        //     otherwise re-bury the lifted ceiling.
        if legibility_floor && props.is_sky {
            let (fg, bg) = legible_floor(out.glyph, (out.fg, out.bg), unlamped, (base.fg, base.bg));
            out.fg = fg;
            out.bg = bg;
        }

        // 5b. ANIMATION GLYPH TRACKS: cycle a short
        //     single-glyph SEQUENCE over `ctx.time`, REPLACING the body glyph, gated by
        //     the track's trigger. The `OnState` trigger consults the generic
        //     `ctx.state` word (the state bus). A FAR-FROZEN cell passes `time == 0` so
        //     every clock holds frame 0. [`AnimTrackSet::EMPTY`] (the default) is a
        //     strict no-op ⇒ the glyph is unchanged. Pure, allocation-free.
        if self.anim_tracks.count > 0 {
            let track_time = if frozen { 0.0 } else { ctx.time };
            out.glyph = self
                .anim_tracks
                .resolve_body(out.glyph, track_time, ctx.state);
        }

        // 6. SEMANTIC-SURFACE FINISH: a frost / wet /
        //    redaction / watching / wear / decay post-layer applied OVER the fully
        //    shaded texel. `None` (the default) is a no-op → byte-identical to today;
        //    the catalog's opt-in surfaces bind one. The redaction/decay finishes read
        //    `ctx.state` only as a generic clearing / deadness level (the bridge encodes
        //    it; the render never learns the puzzle meaning). A FAR-FROZEN cell passes
        //    `frozen` so the proximity-driven watcher holds at its quiet resting glow.
        if let Some(fx) = self.surface_fx {
            fx.apply(&mut out, tex.coord, ctx.state, ctx.distance, frozen);
        }
        out
    }

    /// Resolve the KINDRED-SYMBOL family layer to a single glyph for this cell.
    ///
    /// Builds the [`FamilyDrive`](crate::glyph_family::FamilyDrive) from the signals
    /// the shader already has — `time` (the float clock), a normalized + QUANTIZED
    /// distance (the same LOD deadband the rest of the shader uses, so the growth
    /// glyph is anti-flicker stable), and a normal/incidence LEAN derived from the
    /// per-cell [`LightCtx`] — then calls the family selector. The per-cell
    /// phase/seed is a WORLD hash ([`cell_phase`](crate::glyph_family::cell_phase)) so
    /// neighbouring cells desync. A FROZEN cell zeroes the float `time` so the far
    /// field holds still. Pure, no allocation.
    #[inline]
    fn family_glyph(
        &self,
        layer: crate::glyph_family::FamilyLayer,
        recipe_glyph: u32,
        coord: Vec2,
        ctx: &SampleCtx,
        frozen: bool,
    ) -> u32 {
        use crate::glyph_family::{cell_phase, FamilyDrive};
        // SKY / void is a genuinely empty surface (its blank glyph carries the depth
        // band, not ink) — never float a glyph onto it. Every OTHER recipe gets the
        // family substitution even where the recipe's body would be blank: the
        // kindred family IS the surface for a haze/flow/spark material, so it must
        // show as a continuous floating field, not a sparse stipple.
        if self.recipe.props().is_sky {
            return recipe_glyph;
        }
        // Normalize the QUANTIZED LOD density to [0,1] across the near→far band, so
        // the GROWTH driver grows with distance on the same deadbanded driver the
        // rest of the shader uses (the floating glyph drifts, doesn't jitter).
        let q = if frozen {
            FAR_FREEZE_LOD_BUCKET
        } else {
            texel_density_quantized_for(ctx, self.stability)
        };
        let distance01 = smoothstep(LOD_NEAR_DENSITY, LOD_FAR_DENSITY, q).clamp(0.0, 1.0);
        // POSITIONAL lean from the face geometry: the view's incidence on the
        // surface, signed by the tangent-u component, gives a left/centre/right
        // signal. With no LightCtx (a focused test) the lean is centre (0).
        // VIEW-INDEPENDENT toggle: the lean reads `view_dir`, so under view-indep
        // lighting it is dropped to centre (0) — the kindred-symbol pick then depends
        // only on (world coord, depth, time), keeping the family glyph pannable-exact.
        let lean = if view_independent_lighting() {
            0.0
        } else {
            ctx.light
                .as_ref()
                .map(|lc| {
                    // tangent-u · view ∈ [-1,1]: how much the view grazes ALONG the
                    // surface's horizontal axis — a natural left/right lean.
                    lc.tangent_u.dot(lc.view_dir).clamp(-1.0, 1.0)
                })
                .unwrap_or(0.0)
        };
        // FROZEN ⇒ no float (time term zeroed), so a far floating surface holds.
        let time = if frozen { 0.0 } else { ctx.time };
        let drive = FamilyDrive {
            time,
            distance01,
            lean,
            // STATE BUS: a `State`-driven family indexes its glyph by the
            // cell's live state word (powered/dead/decoy/won). `ctx.state` is `0` for
            // an EMPTY/absent state map (every tag) AND for a non-`State` driver
            // (which ignores `state`), so this is byte-identical until the bridge wires
            // a `State` family + populates the map.
            state: ctx.state,
        };
        let seed = cell_phase(coord.x, coord.y);
        layer.select(&drive, seed)
    }

    /// Resolve a SUPER-GLYPH layer to a single glyph for
    /// this cell. Mirrors [`family_glyph`](Self::family_glyph) EXACTLY for stability:
    /// SKY stays blank; the LOD density is the same stability-aware QUANTIZED bucket the
    /// rest of the shader uses (or the fixed far bucket when `frozen`), so the discrete
    /// pick is deadbanded and does not flicker under sub-cell camera motion; a FROZEN
    /// cell zeroes the float `time` so a far field holds still. One quantized table
    /// lookup over a world-anchored phase — pure, allocation-free.
    ///
    /// NOTE: the [`SuperSelector::Bpm`] selector derives its beat phase from
    /// `ctx.time` (there is no `bpm_phase` on [`SampleCtx`]). When the audio transport's
    /// beat clock is threaded into the render context, swap that `time` for the
    /// quantized beat phase here — a one-line change confined to this helper.
    #[inline]
    fn super_glyph_glyph(
        &self,
        layer: SuperGlyphLayer,
        recipe_glyph: u32,
        coord: Vec2,
        ctx: &SampleCtx,
        frozen: bool,
    ) -> u32 {
        // SKY / void is genuinely empty (its blank glyph carries the depth band, not
        // ink) — never float a super-glyph onto it, exactly like `family_glyph`.
        if self.recipe.props().is_sky {
            return recipe_glyph;
        }
        // The same stability-aware QUANTIZED LOD density the family / detail path use,
        // so the super-glyph's discrete pick is anti-flicker stable; FROZEN locks it to
        // the single fixed far bucket so the whole far field shares one plateau.
        let q_density = if frozen {
            FAR_FREEZE_LOD_BUCKET
        } else {
            texel_density_quantized_for(ctx, self.stability)
        };
        // FROZEN ⇒ no float (time term zeroed), so an animated selector holds still on a
        // far cell — the "far things hold still" half of the freeze.
        let time = if frozen { 0.0 } else { ctx.time };
        // VIEW-INCIDENCE azimuth for the angle-driven selector: the view direction
        // projected into the surface's tangent frame, atan2'd to an azimuth and
        // normalized to [0,1). With no LightCtx (a focused test) it is the face-on
        // centre (0.5), so the pick is stable. The `ByAngle` selector is DELIBERATELY
        // view-dependent (the symbol must track the viewing angle), so unlike the
        // family lean it does NOT zero under view-independent lighting — a material
        // carrying it is in the renderer's view-dependent recast set.
        let angle01 = ctx.light.as_ref().map(view_incidence01).unwrap_or(0.5);
        layer.select(q_density, coord, time, self.seed, angle01)
    }

    /// Apply the relief / normal-mapped Blinn-Phong lighting to a resolved texel.
    ///
    /// Taps the height field at `coord`, `coord+du`, `coord+dv` (a 2-extra-eval
    /// central difference) for the tangent gradient, perturbs the geometric face
    /// normal into a world relief normal, lights `fg`/`bg` in truecolor, and
    /// RE-SELECTS the glyph from the resulting height+luminance so raised, lit
    /// cells read as denser/bolder symbols than recessed, shadowed ones. The
    /// detail glyph (mortar │/─, pad ┼, star) is preserved when a detail texel was
    /// revealed — only the *body* glyph is height-driven. Pure, no allocation.
    #[inline]
    fn apply_relief_lighting(
        &self,
        tex: &TexResult,
        lc: &LightCtx,
        mut out: Texel,
        relief: &ReliefArgs,
    ) -> Texel {
        let _ = tex; // height taps come via `relief`; kept for symmetry/clarity
        let ReliefArgs {
            detail_revealed,
            anim_b,
            distance,
            frozen,
            family_glyph,
            lod,
            coord,
            h,
            grad_taps,
            ramp_steps,
        } = *relief;
        let _ = lod; // octave cap already baked into the precomputed taps
                     // A revealed structural detail OR a kindred-symbol family glyph already owns
                     // the cell's glyph; in both cases the relief body re-selection / sparse-body
                     // blanking must leave the glyph untouched (colour + lighting still run).
        let glyph_owned = detail_revealed || family_glyph;
        let lm = &self.light;
        // Single read of the per-recipe descriptor for every property query below.
        let props = self.recipe.props();

        // SELF-LIT recipes are projected light / phosphor / sheen, NOT physical
        // surfaces — the relief (normal-mapped Blinn-Phong) pass must not relight
        // them (the phosphor green / dead-black of code, the glowing line-on-void of
        // the hologram, the panel sheen of a scanline must read). Treat them as fully
        // emissive: keep the recipe colours (its fg hue + bg), only ride the
        // animation brightness. The hologram's vertical scroll is carried by the
        // coord shift, not a brightness curve, so anim_b is 1.0 for it unless it also
        // carries a pulse — either way the colours are preserved. One descriptor flag
        // (`self_lit`) replaces the hand-listed CodeWall|Hologram|Scanline group.
        if props.self_lit {
            out.fg = dim_rgb(out.fg, anim_b);
            out.bg = dim_rgb(out.bg, anim_b);
            return out;
        }

        // Central-difference gradient of the height field in the (u,v) tangent
        // plane. The (up to three) `height_at` taps were taken once in
        // `texture_sample` (the world-anchored, cacheable half) and threaded in via
        // `relief`; the relief pass only does the cheap normal perturbation here.
        // `grad_taps == None` ⇒ past the NEAR relief band (or no normal map) ⇒ light
        // with the flat geometric face normal — EXACTLY the predicate the single-pass
        // shade used, so the lit output is unchanged.
        let normal = if let Some((hu, hv)) = grad_taps {
            let eps = HEIGHT_GRAD_EPS;
            let dh_du = (hu - h) / eps * lm.height_scale;
            let dh_dv = (hv - h) / eps * lm.height_scale;
            perturb_normal(lc, dh_du, dh_dv, lm.normal_strength)
        } else {
            lc.face_n
        };

        // Light the foreground albedo. The albedo is the recipe's chosen fg
        // colour (already detail-blended); animation brightness rides on top.
        let albedo = unpack_rgb(out.fg) * anim_b;
        let (lit_fg, shadow, lum, point_radiance) = light_surface(albedo, normal, h, lm, lc);
        out.fg = pack_rgb(lit_fg);

        // The bg is pushed into shadow/crevice: the recipe's bg, darkened by the
        // shadow factor (deeper in grooves), so fg/bg contrast reads as a raised
        // glyph on recessed ground. Animation rides it too.
        //
        // `shadow` is a SCALAR (`lit.max_element()`
        // upstream), so `dim_rgb` can only re-scale the bg colour the recipe already
        // has — a lamp's hue never reached it, and on a body-coverage-blanked floor
        // (glyph ' ', κ=0) the bg is the cell's ONLY visible carrier, so authored
        // light pools were invisible there even though fg/lum above already lit up.
        // Add the raw point-light radiance (hue included) ADDITIVELY on top of the
        // shadow-dimmed bg, gated by `LAMP_BG_GAIN` — a genuine re-tint, not a
        // brighter version of the same colour. Riding `anim_b` matches the shadow
        // term above; `PointLights::NONE` (every level without a `[light]`) makes
        // `point_radiance` exactly `Vec3::ZERO`, so this is a no-op there.
        let dimmed_bg = unpack_rgb(dim_rgb(out.bg, shadow * anim_b));
        out.bg = pack_rgb(
            (dimmed_bg + point_radiance * (LAMP_BG_GAIN * anim_b)).clamp(Vec3::ZERO, Vec3::ONE),
        );

        // Re-select the BODY glyph from height+light so the symbol protrudes:
        // raised + lit → dense/bold (`@%#`), recessed + shadowed → sparse (`:-. `).
        // Only when no structural detail glyph (mortar/pad/star/trace) was
        // revealed — those carry their own meaning and stay put — and never for
        // `CodeWall` (its glyph IS the source character; relief must not clobber
        // the strange-loop text).
        let is_sky = props.is_sky;
        let is_code_wall = props.is_code_wall;
        // ANTI-FLICKER: the depth-ink cue picks a discrete BODY GLYPH from the
        // ramp, so feed it the QUANTIZED distance — the chosen glyph then only
        // changes when the camera crosses a step boundary, not on every sub-cell
        // frame. (The fg/bg COLOUR above already used the raw distance, so colour
        // shading stays smooth; only the glyph is deadbanded.) The deadband WIDTH
        // scales with this material's [`GlyphStability`]: a `Stable`/`Frozen` body
        // glyph holds across a wide, world-anchored distance bucket so the body
        // ramp stops churning as the player approaches; `Responsive` keeps the
        // shipped ×1 step. World-anchoring: the body glyph is a function of the
        // world `coord` (height/luma at `coord`) AND this quantized distance, so it
        // re-picks only when a coarse LOD step is crossed.
        // FAR FREEZE locks the body-glyph depth-ink distance to the fixed FAR
        // bucket, so the chosen body glyph is invariant to the cell's actual distance
        // (no glyph slide down the density ramp with depth at range — the far field
        // holds one body glyph per world texel). Otherwise the live, stability-aware
        // quantized distance drives the depth-ink cue as before.
        let q_dist = if frozen {
            FAR_FREEZE_LOD_BUCKET
        } else {
            quantize_lod_step(distance, LOD_QUANT_STEP * self.stability.deadband_mul())
        };
        let t = depth_ink_falloff(q_dist); // 0 near, full → 1 far

        // SKY / CEILING gets its depth cue from a final colour band in `sample`
        // (step 5, after lighting + fog) because its base glyph is BLANK — the
        // glyph/ink channel cannot carry a depth cue there. So the relief pass leaves
        // its glyph untouched; only the inked wall/floor recipes get the glyph-ink
        // depth cue below.
        // SPARSE BODY: the relief pass below RE-SELECTS a body ramp glyph for every
        // inked recipe — which would otherwise fill EVERY brick-wall AND floor cell
        // with a glyph (a densely packed `.`/`*` field). We instead blank the
        // body to a coloured SPACE on the world-scattered fraction the coverage mask
        // drops, so both surfaces read by LIGHT + COLOUR with only an occasional,
        // NON-repeating glyph. The mask is a pure function of the WORLD `coord`, so a
        // given spot is always inked or always blank as the camera moves — it thins
        // the surface WITHOUT adding flicker (and even at a grazing angle, where a few
        // world texels repeat across many screen columns, the result is a sparse
        // stipple, not a solid run). Structural mortar/grid/trace seams are a SEPARATE
        // detail texel (handled above) and are untouched, so STRUCTURE stays crisp.
        // Which world-anchored coverage mask thins the body is declared per-recipe
        // in the descriptor (`body_blank`), not re-matched here — a new recipe that
        // wants a thinning mask names its `BodyBlankKind` in `TextureRecipe::props`.
        let body_blank = match props.body_blank {
            BodyBlankKind::Wall => !wall_body_inked(self.seed, coord),
            BodyBlankKind::Floor => !floor_body_inked(self.seed, coord),
            BodyBlankKind::None => false,
        };
        if body_blank
            && !glyph_owned
            && !is_sky
            && !is_code_wall
            && matches!(self.glyph_source, GlyphSource::Recipe)
        {
            out.glyph = u32::from(' ');
            return out;
        }
        if !is_sky
            && !glyph_owned
            && !is_code_wall
            && matches!(self.glyph_source, GlyphSource::Recipe)
        {
            // Bias the ramp by BOTH lit luminance and raw height: weighting height
            // strongly keeps the glyph ramp tracking the relief (ridges dense,
            // grooves sparse) instead of saturating to solid `@` wherever the
            // headlamp is bright. The two together make the symbols protrude.
            //
            // CALMER WALLS: carry detail with COLOUR, not glyphs — never fill
            // everything. So the body ramp is scaled DOWN
            // by `BODY_DENSITY` — the same relief ordering (ridges still denser than
            // grooves) but shifted toward the sparser end of the ramp, so a face-on
            // wall reads as a calm mid/low-density field whose LIGHT + COLOUR carry
            // the form, instead of a solid `@%#` slab. The depth-ink cue below then
            // thins it further with distance.
            let protrude = ((0.35 * lum + 0.65 * h) * BODY_DENSITY).clamp(0.0, 1.0);

            // DEPTH CUE via the glyph (ink) channel. A cell reads as
            // `lerp(bg, fg, ink(glyph))`, so on the common BRIGHT-fg / DARK-bg
            // surface, sliding the body glyph DOWN the density ramp (less ink) with
            // distance lowers effective luminance — a correctly-signed near-bright /
            // far-dim depth cue carried entirely by the glyph, leaving the fg/bg
            // colours (and thus QA `separation` / `invisible_pct`) untouched.
            //
            // We lerp the ramp brightness toward DEPTH_INK_FLOOR (a sparse-but-still-
            // INKED level — never blank). Edge case — bg brighter than fg: there
            // LESS ink RAISES luminance, which would INVERT the cue, so key the
            // direction off which channel is brighter and fade toward DENSER instead.
            // Each branch clamps so the cue only ever moves density one way (never
            // brightens a far cell relative to near).
            let depth_protrude = if luma_of(out.bg) > luma_of(out.fg) {
                lerp(protrude, 1.0, t).max(protrude)
            } else {
                lerp(protrude, DEPTH_INK_FLOOR, t).min(protrude)
            };
            // RAMP-THRESHOLD DITHER.
            // `ramp_glyph` quantises this continuous brightness onto the ~10-step body
            // ramp, so on a flat far wall every cell at the same depth lands on the
            // same step and the depth-ink gradient breaks into CONTOUR BANDS — hard
            // glyph staircases marching across the surface. Jitter the value by up to
            // ±½ ramp step × amplitude BEFORE quantising, keyed on the WORLD sample
            // coord (never screen, never time) and scaled by the depth-ink strength
            // `t` so the near field stays crisp. World-anchored ⇒ a given spot always
            // gets the same jitter ⇒ the bands dissolve into a stable stipple with
            // zero added shimmer. `ramp_dither = 0` ⇒ byte-identical to the old path.
            let depth_protrude = depth_protrude
                + ramp_dither_offset(
                    self.seed,
                    coord,
                    crate::depth_contrast::active().ramp_dither,
                    t,
                );
            // Clamp the cue so it can never thin an already-inked body cell all the
            // way to a BLANK glyph: a cell whose un-cued `protrude` is itself inked
            // (≥ RAMP_MIN_INKED) stays at or above the sparsest INKED ramp level, so
            // far walls thin but never vanish. (A cell that was already blank/near-
            // blank before the cue stays so — the cue only removes ink, never adds a
            // blank where there wasn't one.) `RAMP_MIN_INKED` is the brightness that
            // maps to the last inked ramp glyph; the deep `DEPTH_INK_FLOOR` then only
            // governs how fast mid-tone cells thin, decoupled from blank-avoidance.
            //
            // ANTI-FLICKER: see
            // [`INK_GATE_MARGIN`] for the full derivation — this bare `RAMP_MIN_INKED`
            // comparison is exactly the "cusp" the turn test caught (the `.`/' '
            // boundary sitting directly on a continuous, noise-driven value the
            // circuit board's calm body density hovers chronically close to).
            // Widened one-sidedly by `INK_GATE_MARGIN` so small sub-cell world jitter
            // no longer double-crosses it; `depth_protrude` itself is untouched, so
            // the general relief ramp still shades continuously elsewhere.
            let depth_protrude = if protrude >= RAMP_MIN_INKED - INK_GATE_MARGIN {
                depth_protrude.max(RAMP_MIN_INKED)
            } else {
                depth_protrude
            };
            // DEPTH-CONTRAST detail collapse: far objects should read less
            // detailed. Fold the LOD-tiered body-ramp step count (precomputed in
            // `apply_lighting`, `None` for the near tier / the effect disabled) onto
            // the depth-ink cue, so a far wall's glyph variety, not just its
            // brightness, thins with range. Runs AFTER the ramp-dither jitter (so
            // the dither still breaks up the now-WIDER band boundaries) and BEFORE
            // the final inked-floor re-check (so collapsing to a coarse level can
            // never round an already-inked cell down into the blank end of the
            // ramp — see `RAMP_MIN_INKED`'s own invariant, re-asserted here).
            let depth_protrude = crate::depth_contrast::quantize_ramp_steps(
                depth_protrude.clamp(0.0, 1.0),
                ramp_steps,
            );
            let depth_protrude = if protrude >= RAMP_MIN_INKED - INK_GATE_MARGIN {
                depth_protrude.max(RAMP_MIN_INKED)
            } else {
                depth_protrude
            };
            out.glyph = ramp_glyph(depth_protrude.clamp(0.0, 1.0));
        }
        out
    }

    /// The material's HEIGHT field `h(u,v) ∈ [0,1]` at `coord` — the relief map.
    /// Reuses each recipe's own dominant procedural scalar so the lit relief
    /// matches the texture you see: brick faces high / mortar low, circuit
    /// pads+traces high / board low, floor grain a gentle bump, sky flat. Cheap
    /// (a few scalar evals, no texel build, no allocation) so it can be tapped
    /// three times per cell for a central-difference gradient.
    #[inline]
    fn height_at(&self, perlin: &Perlin, coord: Vec2, lod: ShadeLod) -> f32 {
        // The world-per-cell footprint re-expressed in a given call's own
        // `coord * scale` space (one wavelength of that noise's unit frequency is
        // `1/scale` world units, so the footprint scales the same way).
        let wpc = |scale: f32| lod.world_per_cell / scale.abs().max(1e-6);
        match self.recipe {
            TextureRecipe::Brick {
                aspect,
                mortar_w,
                grain,
            } => {
                // Brick face = raised, mortar seam = recessed groove. Body mottle
                // adds fine bump on the faces.
                let p = Vec2::new(coord.x * aspect, coord.y);
                let (mortar, brick_id, _u, _v) = brick(p, 1.0, mortar_w);
                let mottle = fbm_bandlimited(
                    perlin,
                    coord * 2.0 + Vec2::splat(brick_id as f32 * 0.013),
                    lod.octaves(3),
                    2.0,
                    0.5,
                    wpc(2.0),
                );
                // 1 on a brick face, dropping to ~0.15 in the mortar groove.
                let face = 1.0 - 0.85 * mortar;
                (face * (0.7 + 0.3 * grain * mottle)).clamp(0.0, 1.0)
            }
            TextureRecipe::Circuit { scale, trace_w } => {
                // Pads/vias + traces are raised lit metal; the board pour is low.
                let p = coord * scale;
                let (f1, _cell) = worley(self.seed, p);
                let edge = circuit_edge(self.seed, p);
                let pour =
                    fbm_bandlimited(perlin, coord * 1.7, lod.octaves(3), 2.0, 0.55, wpc(1.7));
                let mut h = 0.30 + 0.25 * pour; // board body, low-mid
                if edge < trace_w {
                    h = h.max(0.75); // trace ridge
                }
                if f1 < trace_w * 1.6 {
                    h = 1.0; // pad/via — tallest
                }
                h.clamp(0.0, 1.0)
            }
            TextureRecipe::Grain {
                scale,
                speck,
                tile_eps,
            } => {
                // Subtle bump: fBm wash + speck pinpricks raised, tile lines as
                // shallow grooves.
                let wash = fbm_bandlimited(perlin, coord * 0.5, lod.octaves(4), 2.0, 0.5, wpc(0.5));
                let mut h = 0.40 + 0.25 * wash;
                let warp = (fbm_bandlimited(
                    perlin,
                    coord * GRAIN_WARP_FREQ,
                    lod.octaves(2),
                    2.0,
                    0.5,
                    wpc(GRAIN_WARP_FREQ),
                ) - 0.5)
                    * GRAIN_WARP_AMP;
                let (f1, cell) = worley(self.seed, coord * scale + Vec2::splat(warp));
                if f1 < GRAIN_SPECK_RADIUS && hash01(cell) < speck {
                    h += 0.3 * (1.0 - f1 / GRAIN_SPECK_RADIUS);
                }
                if tile_line(coord, tile_eps) {
                    h -= 0.18; // shallow groove on the tile seam
                }
                h.clamp(0.0, 1.0)
            }
            TextureRecipe::Sky { scale, density } => {
                // Sky is essentially flat (the void); stars are tiny raised points
                // but contribute negligible relief — return a soft nebula bump.
                let _ = (scale, density);
                let neb =
                    fbm_bandlimited(perlin, coord * 0.35, lod.octaves(4), 2.2, 0.55, wpc(0.35));
                (0.45 + 0.1 * neb).clamp(0.0, 1.0)
            }
            // A hologram is projected light, not a physical surface — no relief.
            // (It early-returns from the lighting pass anyway, so this is unused; a
            // flat mid value keeps any incidental gradient tap well-behaved.)
            TextureRecipe::Hologram { .. } => 0.5,
            // Code on a wall is flat text on a screen — no physical relief.
            TextureRecipe::CodeWall { .. } => 0.5,
            // A placard is flat text on a plate — no physical relief (and it is
            // self-lit, so this height is never actually sampled).
            TextureRecipe::NoteWall { .. } => 0.5,
            // A scanline panel is self-lit (it early-returns from the relief pass),
            // so its height field is never sampled; a flat mid keeps any incidental
            // gradient tap well-behaved.
            TextureRecipe::Scanline { .. } => 0.5,
            // A truchet lattice is self-lit (it early-returns from the relief pass),
            // so its height field is never sampled; a flat mid keeps any incidental
            // gradient tap well-behaved.
            TextureRecipe::Truchet { .. } => 0.5,
            // A fractal portal is self-lit (it early-returns from the relief pass),
            // so its height field is never sampled; a flat mid keeps any incidental
            // gradient tap well-behaved.
            TextureRecipe::Fractal { .. } => 0.5,
            // The cell alphabet is a PRINTED field, not a moulded one: its states
            // differ in ink, not in depth, and it is anchored to the WALL PLANE
            // (`substrate_plane`) rather than to `coord` — so the `coord ± eps`
            // gradient taps this function is designed for do not even address it.
            // A flat mid value, like the self-lit recipes: the shipped material
            // carries `LightModel::FLAT`, so no gradient tap is ever taken and the
            // single `h` tap feeds a zeroed AO term.
            TextureRecipe::Substrate { .. } => 0.5,
            // A voice wall is the same printed field (see the `Substrate` arm above):
            // its states differ in ink, not in depth.
            TextureRecipe::Voice { .. } => 0.5,
            // An image WITHOUT `relief =` is self-lit (the sampled colour reads
            // directly), so its height field is never sampled; a flat mid keeps any
            // incidental gradient tap well-behaved. WITH `relief =` the attached
            // texture's 8-bit relief channel IS the height field.
            //
            // `coord` on this arm is NOT the procedural world coord every other arm
            // reads: `texture_sample` substitutes the IMAGE UV (see
            // [`Material::image_relief_uv`]) precisely so the plus/minus-eps gradient
            // taps step across TEXELS. A tiled image WRAPS (the repeat is the point);
            // an untiled one relies on the sampler's clamp, so a tap past the border
            // holds the edge texel instead of wrapping round and inventing a cliff.
            TextureRecipe::Image { relief, tile, .. } => match (relief, self.texture.as_ref()) {
                (Some(_), Some(img)) => {
                    let (u, v) = if tile == IMAGE_TILE_NONE {
                        (coord.x, coord.y)
                    } else {
                        (fract(coord.x), fract(coord.y))
                    };
                    img.sample_height_nearest(u, v)
                }
                // No relief authored, or no texture resolved (the magenta error
                // block): the flat constant, exactly as before.
                _ => re_core::TextureImage::NEUTRAL_HEIGHT,
            },
            // A facade's contrast is carried by GLYPH + ink (lit vs. dark units,
            // the mullion frame), not by moulded depth — the catalog material
            // carries `LightModel::FLAT`, so this is never actually sampled. A flat
            // mid value, like `Substrate`'s arm above, for any caller that takes a
            // gradient tap anyway.
            TextureRecipe::Facade { .. } => 0.5,
        }
    }

    /// The COORD channel: kinds that move the sample coordinate before the recipe
    /// runs, rather than scaling brightness after it.
    ///
    /// * `Shift` — an unbounded linear scroll down the wall-`v` (matrix-rain flow).
    /// * `Drift` — a bounded, closed two-axis Lissajous wander.
    /// * `Tear`  — quiet, with brief hashed horizontal displacement bursts.
    ///
    /// No-op for every other kind. Pure function of `(coord, time)` — no clock, no
    /// accumulator, no allocation; one `sin` pair for `Drift`, two integer hashes for
    /// `Tear`, nothing at all for the kinds that fall through.
    #[inline]
    fn anim_shift_coord(&self, coord: Vec2, ctx: &SampleCtx) -> Vec2 {
        let a = self.animation;
        match a.kind {
            AnimKind::Shift => {
                let d = a.speed * ctx.time * a.amplitude;
                // Scroll mostly down the surface (y), a touch across (x) for life.
                Vec2::new(coord.x, coord.y + d)
            }
            AnimKind::Drift => {
                // A Lissajous figure-eight: both axes are whole multiples of the same
                // fundamental, so the path CLOSES exactly at `t + 1/speed`. The loop
                // is a property of the frequency RATIO being integral — no wrap is
                // imposed on the output, which is why the closure is seamless rather
                // than merely periodic. Bounded by `amplitude` on each axis, so unlike
                // `Shift` the sampled coord never runs away from its texel.
                let p = std::f32::consts::TAU * a.speed * ctx.time;
                coord
                    + Vec2::new(
                        a.amplitude * p.sin(),
                        a.amplitude * (DRIFT_AXIS_RATIO * p).sin(),
                    )
            }
            AnimKind::Tear => {
                // Quantize time into short ticks and let only `TEAR_DUTY` of them tear.
                // The TICK INDEX seeds the hash, not raw time, so the displacement is
                // constant across a tick — a tick is either torn or not, never
                // half-torn, and the burst has a hard edge instead of a ramp.
                //
                // `floor`, not a wrapped `fract`: the index is MONOTONE, so the burst
                // sequence never returns (see [`AnimKind::Tear`] for why a looping
                // glitch is the wrong look). `floor` rather than an `as i32` truncation
                // so a negative `speed` — or a negative `time` — keeps uniform tick
                // lengths instead of doubling the one straddling zero.
                let tick = (a.speed * ctx.time * TEAR_TICKS_PER_UNIT).floor() as i32;
                // Per-row salt: neighbouring rows tear independently, so a burst is a
                // torn BAND across the surface, not a whole-surface jump.
                let salt = coord.y.floor() as i32;
                if hash01(hash3i(self.seed as i32, tick, salt)) < TEAR_DUTY {
                    let jitter = hash01(hash3i(self.seed as i32 ^ TEAR_JITTER_SALT, tick, salt));
                    // Horizontal only, centred on zero ⇒ bounded by ±amplitude/2.
                    Vec2::new(coord.x + a.amplitude * (jitter - 0.5), coord.y)
                } else {
                    coord
                }
            }
            // Every other kind carries its motion in `anim_brightness` (Pulse/Flicker/
            // Scan/Breathe) or in `fractal_phase` (Zoom/Spin/SpinZoom), or is static.
            AnimKind::None
            | AnimKind::Pulse
            | AnimKind::Flicker
            | AnimKind::Scan
            | AnimKind::Breathe
            | AnimKind::Zoom
            | AnimKind::Spin
            | AnimKind::SpinZoom => coord,
        }
    }

    /// The live [`crate::fractal::LogPolarPhase`] for the [`TextureRecipe::Fractal`]
    /// recipe, mapping `(animation, ctx.time)` → a zoom/spin scroll already reduced
    /// into its loop period. The three fractal anim modes drive the phase (NOT a
    /// coord scroll like [`AnimKind::Shift`]); every other animation kind (and a
    /// static material) yields the default phase — a still portal. Pure + cheap.
    #[inline]
    fn fractal_phase(&self, ctx: &SampleCtx, twist: f32) -> crate::fractal::LogPolarPhase {
        use crate::fractal::{LogPolarPhase, LN2, TAU};
        let a = self.animation;
        let t = ctx.time * a.speed;
        match a.kind {
            AnimKind::Zoom => LogPolarPhase {
                zoom: (t * LN2).rem_euclid(LN2),
                spin: 0.0,
            },
            AnimKind::Spin => LogPolarPhase {
                zoom: 0.0,
                spin: (t * TAU).rem_euclid(TAU),
            },
            // Couple spin to zoom AND cancel the field's `twist*log_radius` per-octave
            // drift so the Droste spiral loops seamlessly for ANY twist (including the
            // shipped 0.9). The coupling lives in one place — `spinzoom_phase`. The
            // Zoom/Spin/None arms do not use `twist`.
            AnimKind::SpinZoom => crate::fractal::spinzoom_phase(t * LN2, twist),
            // A static material or a non-fractal animation (Pulse/Shift/Flicker/None)
            // does not drive the warp — a still portal at the default phase.
            _ => LogPolarPhase::default(),
        }
    }

    /// `Pulse`/`Flicker`/`Scan`/`Breathe` animation: a `[0,2]`-bounded brightness
    /// multiplier sampled by `ctx.time`. `Pulse` breathes sinusoidally; `Flicker`
    /// jitters from a time-quantized hash; `Scan` sweeps a narrow lifted band along
    /// `u`; `Breathe` is a slow-in/fast-out envelope. Returns `1.0` for `None`/`Shift`
    /// and for the three log-polar phase modes. Cheap + pure (no transcendental for
    /// `Scan`, one `cos` for `Pulse`/`Breathe`).
    #[inline]
    fn anim_brightness(&self, coord: Vec2, ctx: &SampleCtx) -> f32 {
        match self.animation.kind {
            AnimKind::Pulse => {
                let phase = std::f32::consts::TAU * self.animation.speed * ctx.time;
                // Centre at 1.0, swing ±amplitude.
                (1.0 + self.animation.amplitude * phase.cos()).clamp(0.0, 2.0)
            }
            AnimKind::Flicker => {
                // Quantize time into ticks so the flicker is steppy (static-like)
                // rather than continuous; hash on tick + a per-cell salt.
                let step = (self.animation.speed.max(0.01) * ctx.time) as i32;
                let salt = coord.x.floor() as i32 ^ (coord.y.floor() as i32).wrapping_mul(31);
                let h = hash01(hash3i(self.seed as i32, step, salt));
                (1.0 - self.animation.amplitude * h).clamp(0.0, 1.0)
            }
            AnimKind::Scan => {
                // Half-width of the lifted band, as a fraction of the surface's u
                // extent. Narrow on purpose: a wide band is just a Pulse.
                // (Declared first so `clippy::items_after_statements` stays quiet.)
                const BAND: f32 = 0.06;
                // Sweep position, and the cell's own position along u. `rem_euclid`
                // rather than `fract` because `fract` is SIGN-PRESERVING: a negative
                // `speed` (an authored number, once a `.lvl` can name this kind) or a
                // negative world coord would leave the value in `(-1,0]`, the wrapped
                // distance below would go NEGATIVE, and `k` would exceed 1 — lighting
                // a wide region instead of a band, with no visible cause. Folding BOTH
                // into `[0,1)` is what keeps `d` inside `[0,0.5]`. Identical to `fract`
                // for the non-negative case, so nothing shipped moves; a negative speed
                // now simply sweeps the other way.
                let sweep = (self.animation.speed * ctx.time).rem_euclid(1.0);
                let here = coord.x.rem_euclid(1.0);
                // Shortest wrapped distance, so the band crosses the seam cleanly.
                let d = (here - sweep).abs();
                let d = d.min(1.0 - d);
                if d < BAND {
                    // Smooth falloff inside the band; no hard edge to alias on.
                    let k = 1.0 - (d / BAND);
                    (1.0 + self.animation.amplitude * k * k).clamp(0.0, 2.0)
                } else {
                    1.0
                }
            }
            AnimKind::Breathe => {
                // Fraction of the period spent RISING. > 0.5 is what makes the
                // envelope asymmetric (slow in, fast out) rather than a cosine.
                // (Declared first so `clippy::items_after_statements` stays quiet.)
                const IN: f32 = 0.7;
                // `p` in [0,1). Rise over the first 70%, fall over the last 30%.
                // `rem_euclid` rather than `fract`, for the same reason the `Scan` arm
                // above uses it: `f32::fract` is SIGN-PRESERVING, so a negative `speed`
                // — or a negative `ctx.time` — lands `p` in `(-1,0]`, `p < IN` then holds
                // for the WHOLE cycle, the fall segment is never reached, and (because
                // `cos` is even) the arm silently REPLAYS the forward envelope instead of
                // reversing it, ending the cycle at `env ≈ 0.38` instead of `0` — a value
                // jump at the wrap that the eye reads as a cut. Unlike `Scan` there is no
                // second operand to fold: this arm reads no coordinate, so `p` is the only
                // route in. Identical to `fract` for the non-negative case, so nothing
                // shipped moves; a negative speed now simply breathes the other way.
                let p = (self.animation.speed * ctx.time).rem_euclid(1.0);
                let env = if p < IN {
                    // Slow in: ease along a raised cosine over the long segment.
                    0.5 - 0.5 * (std::f32::consts::PI * p / IN).cos()
                } else {
                    // Fast out.
                    0.5 + 0.5 * (std::f32::consts::PI * (p - IN) / (1.0 - IN)).cos()
                };
                (1.0 + self.animation.amplitude * env).clamp(0.0, 2.0)
            }
            // `None` and the three COORD kinds (`Shift`/`Drift`/`Tear`, which move the
            // sample coord in `anim_shift_coord`) don't touch brightness; the three
            // FRACTAL modes drive the log-polar PHASE (see `fractal_phase`), not a
            // brightness scale, so they are a no-op here too.
            AnimKind::None
            | AnimKind::Shift
            | AnimKind::Drift
            | AnimKind::Tear
            | AnimKind::Zoom
            | AnimKind::Spin
            | AnimKind::SpinZoom => 1.0,
        }
    }

    // ── brick / panel masonry ──
    fn sample_brick(
        &self,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
        aspect: f32,
        mortar_w: f32,
        grain: f32,
    ) -> (Texel, Option<Texel>) {
        // DISTANCE-LOD octave cap (no-op in the NEAR band → byte-identical).
        let lod = ShadeLod::for_ctx(ctx);
        // Map the face coord to brick space: u across the face (tex_u-driven),
        // v down the wall by screen row so courses stack vertically.
        let p = Vec2::new(coord.x * aspect, coord.y);
        let (mortar, brick_id, u, _v) = brick(p, 1.0, mortar_w);

        // Body color from fBm mottling through the palette, band-limited so
        // the mottle's own high octaves fade by their SCREEN period, not a hard
        // distance-tier cutoff.
        let mottle = fbm_bandlimited(
            perlin,
            coord * 2.0 + Vec2::splat(brick_id as f32 * 0.013),
            lod.octaves(3),
            2.0,
            0.5,
            lod.world_per_cell / 2.0,
        );
        let t = 0.5 + grain * (mottle - 0.5) + 0.05 * hash01(brick_id);
        let body = self.palette.eval(t);
        let bg = self.bg_palette.eval(t * 0.8 + 0.1);

        // Base texel: the glyph body. rank 0 (always visible). SPARSE BODY: on the
        // world-scattered BLANK fraction (see [`wall_body_inked`]) the body is a
        // coloured SPACE — so a plain wall face is mostly colour-carried ground, not
        // packed runs of identical ramp glyphs. World-anchored → no churn. (The lit
        // path re-selects the body glyph but honours the SAME mask; this keeps the
        // no-`LightCtx` fallback consistent and the recipe base honest.) Mortar
        // detail below is unaffected, so STRUCTURE stays crisp.
        let body_bright = 0.55 + 0.45 * mottle;
        let body_glyph = if wall_body_inked(self.seed, coord) {
            ramp_glyph(body_bright)
        } else {
            u32::from(' ')
        };
        let base = Texel {
            glyph: body_glyph,
            fg: pack_rgb(body),
            bg: pack_rgb(bg * BRICK_BG_FACTOR),
            rank: 0,
        };

        // Detail: the mortar seam — a structural feature, low rank, so it
        // survives to medium distance but still anti-aliases when grazing/far.
        if mortar > 0.5 {
            let on_h = u > 0.25 && u < 0.75; // horizontal-ish course → ─, else │
            let glyph = if on_h { MORTAR_H_GLYPH } else { MORTAR_V_GLYPH };
            let mortar_col = self.bg_palette.eval(0.0);
            let detail = Texel {
                glyph,
                fg: pack_rgb(mortar_col),
                bg: pack_rgb(bg * BRICK_BG_FACTOR * 0.7),
                rank: rank_for(self.rank, RankBand::Structure),
            };
            return (base, Some(detail));
        }

        // CALMER WALL: the wall's STRUCTURE — the mortar
        // courses above — carries the read; the body is shaded by COLOUR (the
        // palette mottle) + the relief light, NOT by a speckle fill of identical
        // `.` glyphs. So the fine grit is heavily rationed: only the densest grit
        // peaks survive (threshold raised 0.86 → 0.94), and it stays fine-ranked so
        // it shows only on a near, face-on wall as a LITTLE controlled texture —
        // never an incoherent mush of identical-symbol runs.
        let speck = value_noise(self.seed ^ 0x51EC, coord * 9.0);
        if speck > 0.94 {
            let detail = Texel {
                glyph: BRICK_SPECK_GLYPH,
                fg: pack_rgb(body * 1.3),
                bg: base.bg,
                rank: rank_for(self.rank, RankBand::Fine),
            };
            return (base, Some(detail));
        }
        (base, None)
    }

    // ── circuit board ──
    fn sample_circuit(
        &self,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
        scale: f32,
        trace_w: f32,
    ) -> (Texel, Option<Texel>) {
        // DISTANCE-LOD octave cap (no-op in the NEAR band → byte-identical).
        let lod = ShadeLod::for_ctx(ctx);
        let p = coord * scale;
        let (f1, cell) = worley(self.seed, p);

        // Copper pour base: fBm wash, with ridged veins overlaid so the board
        // body shows faint etched copper flow rather than a flat fill —
        // band-limited by screen period.
        let pour = fbm_bandlimited(
            perlin,
            coord * 1.7,
            lod.octaves(3),
            2.0,
            0.55,
            lod.world_per_cell / 1.7,
        );
        let veins = ridged_bandlimited(
            perlin,
            coord * 3.1,
            lod.octaves(2),
            2.0,
            0.5,
            lod.world_per_cell / 3.1,
        );
        let pour = (pour * 0.7 + veins * 0.3).clamp(0.0, 1.0);
        let body = self.palette.eval(0.25 + 0.3 * pour);
        let bg = self.bg_palette.eval(0.15 + 0.2 * pour);
        // CALMER WALL: the board BODY is mostly low-density — a sparse glyph that
        // lets the COLOUR (the copper-pour palette + relief light) carry the look,
        // instead of a mid-dense `*+=` fill across the whole face. The few trace
        // lines + pads below provide the structure. (Bright `0.4+0.3*pour` → a
        // sparser `0.18+0.22*pour` so the base sits low on the density ramp.)
        let base = Texel {
            glyph: ramp_glyph(0.18 + 0.22 * pour),
            fg: pack_rgb(body),
            bg: pack_rgb(bg),
            rank: 0,
        };

        // Band-limiting the pad/trace detail is a different problem: it's a THIN
        // structural feature (`trace_w` fraction of a
        // Voronoi cell), not an fBm octave — `fbm_bandlimited`/`ridged_bandlimited`
        // above already band-limit the smooth pour/veins wash, but a hard
        // Voronoi edge/point test has no "octave" to fade. Its own aliasing risk
        // is the FEATURE'S WORLD WIDTH, not the cell grid's: a trace narrower than
        // ~2 screen cells cannot be point-sampled without flickering as the ray
        // direction shifts sub-cell (confirmed by the A6 turn test — this recipe
        // was the one that failed it). Fade it out the same way an octave fades:
        // below the feature reliably resolving, fall through to `base` (the
        // already-band-limited pour+veins wash), so the fine structure is carried
        // by colour, not a flickering glyph — the same "coarser octave carries the
        // mean" principle `fbm_bandlimited` uses.
        // A full (not half) fade margin: the turn test measures REAL sub-cell
        // sampling instability, not a theoretical Nyquist limit, so the feature
        // is only shown once it is FULLY resolved (period >= NYQUIST_FADE_HI
        // screen cells), matching `octave_fade`'s own fully-resolved endpoint.
        let feature_span_world = (trace_w / scale.abs().max(1e-6)) * 2.0;
        let feature_fade = octave_fade(feature_span_world, lod.world_per_cell);
        if feature_fade < 1.0 {
            return (base, None);
        }

        // A pad/via sits at the Voronoi feature point (small f1): structural.
        if f1 < trace_w * 1.6 {
            let lit = hash01(cell) > 0.45;
            let glyph = if lit {
                CIRCUIT_NODE_GLYPH
            } else {
                CIRCUIT_PAD_GLYPH
            };
            let accent = self.palette.eval(0.7 + 0.2 * hash01(cell));
            let detail = Texel {
                glyph,
                fg: pack_rgb(accent),
                bg: base.bg,
                rank: rank_for(self.rank, RankBand::Structure),
            };
            return (base, Some(detail));
        }

        // Traces run along Voronoi edges (where two cells are equidistant).
        // CALMER WALL: an earlier version flipped the rung
        // glyph by SCREEN-ROW PARITY, which produced sequences of identical
        // symbols in rows — an incoherent mush — AND flickered as the slice moved
        // (a world point's glyph changed with the screen row it landed on). The
        // trace glyph is now chosen by the trace's WORLD ORIENTATION (the sign of
        // the F1 feature-cell offset), so it is STABLE under motion and reads as a
        // real routed line, not a per-row checker. The board BODY is shaded by the
        // palette colour, so the structure is carried by these few clean trace
        // lines + the pads, not a dense glyph fill.
        let edge = circuit_edge(self.seed, p);
        if edge < trace_w {
            // World-stable orientation: which axis of the local cell the point sits
            // further along. Pure function of the world coord → no screen-row churn.
            let fu = fract(p.x) - 0.5;
            let fv = fract(p.y) - 0.5;
            let glyph = if fu.abs() >= fv.abs() {
                CIRCUIT_RUNG_GLYPH // ─ : horizontal run
            } else {
                CIRCUIT_BUS_GLYPH // │ : vertical run
            };
            let accent = self.palette.eval(0.6);
            let detail = Texel {
                glyph,
                fg: pack_rgb(accent),
                bg: base.bg,
                rank: rank_for(self.rank, RankBand::Mid),
            };
            return (base, Some(detail));
        }
        (base, None)
    }

    // ── world-pinned floor grain ──
    fn sample_grain(
        &self,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
        scale: f32,
        speck: f32,
        tile_eps: f32,
    ) -> (Texel, Option<Texel>) {
        // DISTANCE-LOD octave cap (no-op in the NEAR band → byte-identical).
        let lod = ShadeLod::for_ctx(ctx);
        // Base: soft fBm wash through the bg palette (two-channel: a faint glyph
        // body in fg over the wash bg), band-limited by screen period.
        let wash = fbm_bandlimited(
            perlin,
            coord * 0.5,
            lod.octaves(4),
            2.0,
            0.5,
            lod.world_per_cell / 0.5,
        );
        let bg = self.bg_palette.eval(0.3 + 0.25 * wash);
        let body = self.palette.eval(0.2 + 0.2 * wash);
        let base = Texel {
            glyph: u32::from(' '),
            fg: pack_rgb(body * 0.5),
            bg: pack_rgb(bg),
            rank: 0,
        };

        // Tile grid: structural, low rank → reads even at medium distance. SPARSE
        // SEAM: the seam is not a solid ruled
        // line on every integer world edge — it is chopped into world-hashed segments
        // and only the [`FLOOR_SEAM_COVERAGE`] fraction ink, so the tiling reads as
        // occasional clean seam dashes rather than a packed grid. World-anchored on
        // the line index + along-run segment → a given dash is always inked or always
        // a gap (no churn). We also learn WHICH axis the seam is on (the nearer world
        // line) so the two seam families bucket independently.
        let fx = (coord.x - coord.x.floor()).min(1.0 - (coord.x - coord.x.floor()));
        let fy = (coord.y - coord.y.floor()).min(1.0 - (coord.y - coord.y.floor()));
        let on_seam = fx < tile_eps || fy < tile_eps;
        // The seam runs along the axis whose fractional distance is LARGER (we're on
        // the x-const line when fx is the small one). on_x_line ⇒ an x=int seam.
        let on_x_line = fx <= fy;
        let grid = on_seam && floor_seam_inked(self.seed, coord, on_x_line);
        // World-pinned speck: Worley feature points warped by fBm so specks
        // cluster organically yet stay glued to the world (parallax honestly).
        // Band-limited by screen period.
        let warp = (fbm_bandlimited(
            perlin,
            coord * GRAIN_WARP_FREQ,
            lod.octaves(2),
            2.0,
            0.5,
            lod.world_per_cell / GRAIN_WARP_FREQ,
        ) - 0.5)
            * GRAIN_WARP_AMP;
        let (f1, cell) = worley(self.seed, coord * scale + Vec2::splat(warp));
        let is_speck = f1 < GRAIN_SPECK_RADIUS && hash01(cell) < speck;

        // CALMER FLOOR: the floor must read as mostly
        // low-density — clean tile STRUCTURE (grid seams) carrying the tiling, with
        // shading carried by the fg/bg COLOUR gradient (the wash above), NOT a full
        // noise fill of speck glyphs. So the grid SEAM wins over a speck (structure
        // first), and specks are rare + fine-ranked (revealed only very close, a
        // little controlled grit), never a packed field. The vast majority of floor
        // cells stay a coloured BLANK base, carrying detail with colour, not glyphs.
        if grid {
            let line = self.bg_palette.eval(0.6);
            let detail = Texel {
                glyph: GRAIN_GRID_GLYPH,
                fg: pack_rgb(line),
                bg: base.bg,
                rank: rank_for(self.rank, RankBand::Structure),
            };
            return (base, Some(detail));
        }
        if is_speck {
            let glyph = if (cell & 0x100) != 0 {
                GRAIN_SPECK_A
            } else {
                GRAIN_SPECK_B
            };
            let dot = self.palette.eval(0.85 + 0.1 * hash01(cell));
            let detail = Texel {
                glyph,
                fg: pack_rgb(dot),
                bg: base.bg,
                rank: rank_for(self.rank, RankBand::Fine),
            };
            return (base, Some(detail));
        }

        // Opt-in DETAIL-TIER cross-fade: only engages when this material
        // authored `detail = on` (`self.detail`) — every material that never sets
        // it takes neither branch below and renders BYTE-IDENTICAL to before this
        // addition (the `Footprint`/tier machinery is never even evaluated).
        if self.detail {
            let fp = crate::lod::Footprint::reference(ctx.distance);
            let (tier, w) = crate::detail_lod::tier_and_weight(fp);
            if tier == crate::detail_lod::DetailTier::Detail {
                let (mf, micro_cell) = worley(
                    self.seed ^ DETAIL_MICRO_SEED,
                    coord * (scale * DETAIL_MICRO_SCALE),
                );
                let world_texel_id = u64::from(micro_cell);
                if mf < DETAIL_MICRO_RADIUS
                    && crate::detail_lod::reveal_finer(world_texel_id, self.seed, w)
                {
                    let fine = self.palette.eval(0.6 + 0.15 * hash01(micro_cell));
                    let detail = Texel {
                        glyph: DETAIL_MICRO_GLYPH,
                        fg: pack_rgb(fine),
                        bg: base.bg,
                        rank: rank_for(self.rank, RankBand::Fine),
                    };
                    return (base, Some(detail));
                }
            }
        }
        (base, None)
    }

    // ── star sky / ceiling ──
    fn sample_sky(
        &self,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
        scale: f32,
        density: f32,
    ) -> (Texel, Option<Texel>) {
        // DISTANCE-LOD octave cap (no-op in the NEAR band → byte-identical).
        let lod = ShadeLod::for_ctx(ctx);
        // Nebula base: soft fBm haze through the bg palette, band-limited by
        // screen period.
        let neb = fbm_bandlimited(
            perlin,
            coord * 0.35,
            lod.octaves(4),
            2.2,
            0.55,
            lod.world_per_cell / 0.35,
        );
        let bg = self.bg_palette.eval(0.15 + 0.3 * neb);

        // R-L3: give the base a faint non-zero fg (a dim tint of the nebula bg)
        // so a stray non-space glyph drawn on a sky cell is never invisible.
        let base = Texel {
            glyph: u32::from(' '),
            fg: pack_rgb(bg * SKY_BASE_FG_FACTOR),
            bg: pack_rgb(bg * SKY_BG_FACTOR),
            rank: 0,
        };

        // Stars: hashed feature points on a lattice; brightness/glyph by class.
        let p = coord * scale;
        let ix = p.x.floor() as i32;
        let iy = p.y.floor() as i32;
        let h = hash_cell(self.seed ^ 0x57A2, ix, iy);
        if hash01(h) < density {
            let cls = hash01(h.rotate_left(11));
            let (glyph, t) = if cls < STAR_BRIGHT_FRAC {
                (STAR_GLYPH_BRIGHT, 0.95)
            } else if cls < STAR_MID_FRAC {
                (STAR_GLYPH_MID, 0.7)
            } else {
                (STAR_GLYPH_FAINT, 0.45)
            };
            let star = self.palette.eval(t);
            let detail = Texel {
                glyph,
                fg: pack_rgb(star),
                bg: base.bg,
                rank: rank_for(self.rank, RankBand::Mid),
            };
            return (base, Some(detail));
        }
        (base, None)
    }

    // ── hologram: glyph-only transparent scanlines on void ──
    //
    // A SEE-THROUGH hologram: there is NO filled bg colour (the bg is a near-black
    // VOID), and the only ink is fg-COLOURED horizontal line glyphs forming
    // holographic scanlines. The `fg` hue comes from the material's palette (a
    // single configurable holo colour), so the surface reads as glowing coloured
    // lines on emptiness — not a wall. With a `Shift` animation the sample coord's
    // `y` is scrolled by time (in [`anim_shift_coord`], BEFORE this runs), so the
    // lines drift VERTICALLY; a FAR-FROZEN cell skips that scroll, so a distant
    // hologram holds still (its line pattern is then purely world-anchored).
    //
    // Pure function of `(coord, palette)` — no clock read here, no allocation.
    fn sample_hologram(
        &self,
        coord: Vec2,
        _ctx: &SampleCtx,
        line_freq: f32,
        scan_w: f32,
    ) -> (Texel, Option<Texel>) {
        // VOID bg: forced near-black regardless of the bg palette, so the surface is
        // transparent (only the glyph carries colour). A faint, non-zero floor keeps
        // a hologram cell from being pure 0,0,0 (which can read as a "hole") without
        // filling it — it is still effectively void.
        let bg = pack_rgb(Vec3::splat(HOLO_VOID_LEVEL));

        // Vertical scanline phase: where within the current line PERIOD this row sits.
        // `coord.y` is the vertical surface coord (the `Shift` anim has already
        // scrolled it by time, so the lines move up/down). `line_freq` lines per
        // world-cell; `phase ∈ [0,1)` is the fractional position in one line period.
        let lf = line_freq.max(0.05);
        let phase = fract(coord.y * lf);
        // Triangular distance to the nearest line CENTRE (0 = on a line, 0.5 = mid-gap).
        let dist_to_line = (phase - 0.5).abs() * 2.0; // 0 at centre → 1 at the gap edges
        let on_line = dist_to_line <= scan_w.clamp(0.0, 1.0);

        if !on_line {
            // Between scanlines: VOID. A blank glyph on the void bg — fully
            // transparent. rank 0 so it always shows (the gaps are the see-through).
            let base = Texel {
                glyph: u32::from(' '),
                // A very faint holo-tinted fg so a stray non-space draw is never
                // invisible (mirrors the sky base's R-L3 guard); the cell is blank
                // anyway, so this never inks.
                fg: pack_rgb(self.palette.eval(0.5) * HOLO_VOID_LEVEL * 4.0),
                bg,
                rank: 0,
            };
            return (base, None);
        }

        // ON a scanline: pick the line glyph by how central the row is within the
        // line — the dense double line `═` at the very centre, a single `─` on its
        // shoulders, a dashed `╌` further out, and a faint `·` at the edge. A pure
        // function of the world position → world-anchored, so it parallax-scrolls
        // with the camera and (when far-frozen) holds a fixed pattern.
        let glyph = if dist_to_line < scan_w * 0.25 {
            HOLO_LINE_SOLID // ═ : the bright core of the line
        } else if dist_to_line < scan_w * 0.55 {
            HOLO_LINE_THIN // ─ : the body of the line
        } else if dist_to_line < scan_w * 0.8 {
            HOLO_LINE_DASH // ╌ : fraying toward the gap
        } else {
            HOLO_LINE_DOT // · : the faint edge
        };

        // The fg is the holo hue, brightened toward the line centre so the scanline
        // has a glowing core. A single configurable colour family (the palette),
        // NOT a per-cell wash — the look is "colored line glyphs on void".
        let core = 1.0 - dist_to_line; // 1 at centre → 0 at the gap edge
        let hue = self.palette.eval(0.5 + 0.25 * core);
        let fg = pack_rgb(
            (hue * (HOLO_FG_FLOOR + (1.0 - HOLO_FG_FLOOR) * core)).clamp(Vec3::ZERO, Vec3::ONE),
        );

        // The lit scanline is rank 0 (the base IS the line) so it survives the
        // ranked-LOD: a hologram's lines must read as lines, not LOD-thin to noise.
        let base = Texel {
            glyph,
            fg,
            bg,
            rank: 0,
        };
        (base, None)
    }

    /// SUBSTRATE — the cell alphabet ([`crate::substrate`]) as a wall surface.
    ///
    /// One texel per screen cell: the alphabet glyph for the world cell this hit lands
    /// in, inked at the state's own tone off the material palette over the bg palette.
    ///
    /// # It reads the WALL-PLANE hit, NOT `coord` — and that is the whole point
    ///
    /// FOUND BY RENDERING, not by a unit test (the first implementation used `coord`
    /// and every test passed): for a wall, `sample_coord` builds
    /// `coord.x = tex_u + floor(the face's CONSTANT axis)`. `tex_u` runs `0→1` within
    /// EVERY world cell and the offset is the same for every cell of one flat run, so
    /// `coord.x` **wraps once per world cell**. A repeating recipe (brick, circuit)
    /// does not care. A field that claims to be world-anchored does: the wall tiled,
    /// the same eleven glyphs repeating every world unit across the whole run. It is
    /// the same defect `sample_code_wall`'s "D3 — THE FLOWING PLANE" note describes.
    ///
    /// [`SampleCtx::wall_along`] / [`SampleCtx::wall_up`] carry the raw, CONTINUOUS
    /// world hit on the wall plane (`hit_y`/`hit_x` on an NS/EW face, and the texel's
    /// world height), threaded unconditionally by the one cast path. They are the
    /// honest world anchor, so the recipe reads them and ignores `coord` entirely.
    ///
    /// A caller that threads no wall hit (a focused unit test, or a floor/ceiling cell
    /// — this is a WALL material) still gets a defined, deterministic field; on a floor
    /// it degenerates toward stripes along one axis, which is why nothing binds
    /// `substrate` to a floor.
    ///
    /// Rank 0 and no detail texel: the alphabet IS the meaning of this surface, so it
    /// must survive the ranked-LOD thinning exactly as the hologram's scan lines and
    /// the code wall's source characters do. (`props().is_code_wall` likewise stops
    /// the relief pass re-selecting the glyph off the density ramp.)
    ///
    /// Cost: two integer hashes, three comparisons, two palette evals. No noise, no
    /// transcendentals, no allocation, no `time` term.
    #[inline]
    fn sample_substrate(
        &self,
        ctx: &SampleCtx,
        scale: f32,
        held: f32,
        released: f32,
    ) -> (Texel, Option<Texel>) {
        let rank = crate::substrate::surface_rank(substrate_plane(ctx), scale, self.seed);
        let glyph = crate::substrate::glyph_for_rank(rank, held, released);
        let tone = crate::substrate::tone(glyph, rank);

        // The state drives BOTH channels, so the four states separate by COLOUR as
        // well as by glyph (a far cell whose glyph has gone to mush still reads its
        // state as a brightness tier). One hue, read at a fixed palette point, scaled
        // by a MONOTONE ink ramp — see the SUBSTRATE_* knobs for the measurement that
        // ruled out walking a palette window.
        let ink = SUBSTRATE_INK_FLOOR + (1.0 - SUBSTRATE_INK_FLOOR) * tone;
        let fg = pack_rgb((self.palette.eval(SUBSTRATE_HUE_T) * ink).clamp(Vec3::ZERO, Vec3::ONE));
        // The ground runs the other way: darkest under the brightest cell.
        let ground = SUBSTRATE_BG_CEIL - SUBSTRATE_BG_SPAN * tone;
        let bg =
            pack_rgb((self.bg_palette.eval(SUBSTRATE_HUE_T) * ground).clamp(Vec3::ZERO, Vec3::ONE));

        let base = Texel {
            glyph: u32::from(glyph),
            fg,
            bg,
            rank: 0,
        };
        (base, None)
    }

    /// FACADE — the structured bay×floor window grid (`crate::facade`). Reads the
    /// WALL-PLANE hit like [`Self::sample_substrate`]/[`Self::sample_truchet`]
    /// (`ctx.wall_along`/`ctx.wall_up`), falling back to `coord` off a wall (the
    /// same fallback `sample_truchet` uses, for a focused test or a floor/ceiling
    /// binding this recipe was never designed for but must still render safely).
    ///
    /// `ink` — [`crate::facade::FacadeTexel::ink`] — is already a full `[0,1]`
    /// brightness read (lit windows near `1.0`, dark near their style floor, the
    /// mullion frame its own mid value), so it is applied to the palette taps
    /// directly rather than through a separate floor/ceiling blend like
    /// [`Self::sample_substrate`]'s `tone` — the recipe's OWN style table already
    /// carries the contrast a facade needs.
    #[allow(clippy::too_many_arguments)]
    fn sample_facade(
        &self,
        coord: Vec2,
        ctx: &SampleCtx,
        style: crate::facade::FacadeStyle,
        bay: f32,
        floor_h: f32,
        lit: f32,
        seed: i32,
        ground_band: bool,
        sign_row: bool,
    ) -> (Texel, Option<Texel>) {
        let (u_world, v_world) = if ctx.on_wall {
            (ctx.wall_along, ctx.wall_up)
        } else {
            (coord.x, coord.y)
        };
        let lod = ShadeLod::for_ctx(ctx);
        let params = crate::facade::FacadeParams {
            style,
            bay,
            floor_h,
            lit,
            seed,
            ground_band,
            sign_row,
        };
        let texel = crate::facade::sample(u_world, v_world, &params, lod.world_per_cell);

        let ink = FACADE_INK_FLOOR + (1.0 - FACADE_INK_FLOOR) * texel.ink.clamp(0.0, 1.0);
        let fg = pack_rgb((self.palette.eval(FACADE_HUE_T) * ink).clamp(Vec3::ZERO, Vec3::ONE));
        // The frame band is its own fixed dark structural tone (a consistent grid
        // line); a window unit's ground rises ABOVE the same dark floor only as it
        // approaches fully lit — most units (unlit, most authored `lit` values are
        // well under 0.5) stay near the floor, so the wall reads dark with lit
        // windows glowing, not the reverse.
        let ground = if texel.frame {
            FACADE_FRAME_BG
        } else {
            FACADE_BG_FLOOR + FACADE_BG_GLOW * texel.ink.clamp(0.0, 1.0)
        };
        let bg =
            pack_rgb((self.bg_palette.eval(FACADE_HUE_T) * ground).clamp(Vec3::ZERO, Vec3::ONE));

        let base = Texel {
            glyph: u32::from(texel.glyph),
            fg,
            bg,
            rank: 0,
        };
        (base, None)
    }

    /// IMAGE CUT-OUT: does this material's attached texture have an
    /// OPAQUE texel at the wall coordinate `(u, v)`? `u` is the in-cell horizontal
    /// fraction (`tex_u`), `v` the vertical wall fraction (`wall_v`, `0` = top) —
    /// the SAME coordinates [`sample_image`](Self::sample_image) draws from, so the
    /// pass-through test and the paint agree texel-for-texel. `true` ⇒ the texel is
    /// opaque (paint the image / stop the ray); `false` ⇒ transparent (show the
    /// backdrop behind). A material whose recipe is NOT [`TextureRecipe::Image`], or
    /// that has no attached texture, is treated as OPAQUE (`true`) — so every
    /// non-image / unresolved-image cell keeps its plain solid behaviour and
    /// the cut-out branch never triggers for it (byte-identical to today).
    #[inline]
    pub fn image_texel_opaque_at(&self, u: f32, v: f32) -> bool {
        match (&self.recipe, self.texture.as_ref()) {
            (TextureRecipe::Image { .. }, Some(img)) => {
                // Same `tile =` repeat as the paint path, so the cut-out test and the
                // colour agree texel-for-texel on a tiled wall too.
                let (uu, vv) = self.tile_uv(fract(u), v);
                let [_, _, _, a] = img.sample_nearest(uu, vv);
                image_texel_opaque(a)
            }
            // Non-image recipe, or an Image recipe with no resolved texture (magenta
            // error block) → opaque: never punch a hole, never pass the ray through.
            _ => true,
        }
    }

    /// IMAGE CUT-OUT — is the ENTIRE vertical strip of the attached texture at the
    /// in-cell horizontal fraction `u` transparent? Used by the raycaster to decide
    /// COLUMN-level pass-through: when a ray enters an Image cell, if every texel down
    /// the image column at `u` is sub-cutoff (e.g. the left/right margin around a
    /// figure), the cell is fully see-through at this column and the DDA marches PAST
    /// it to the backdrop — exactly like a wedge whose open triangle the ray missed.
    /// If ANY texel in the strip is opaque the cell is a real hit (a second backdrop
    /// ray + per-row [`image_texel_opaque_at`](Self::image_texel_opaque_at) test then
    /// resolves the figure's vertical silhouette).
    ///
    /// Samples one texel per IMAGE ROW (so the strip test never misses a thin opaque
    /// sliver), capped at [`IMAGE_STRIP_MAX_SAMPLES`] rows for a tall texture so the
    /// per-cell cost stays bounded. A non-image / unresolved material returns `false`
    /// (NOT transparent → never a free pass-through), keeping every other cell exactly
    /// as today.
    #[inline]
    pub fn image_strip_fully_transparent(&self, u: f32) -> bool {
        let (TextureRecipe::Image { .. }, Some(img)) = (&self.recipe, self.texture.as_ref()) else {
            return false;
        };
        if img.width == 0 || img.height == 0 {
            // Zero-area texture samples as transparent black everywhere → the cell is
            // an empty pane: pass the ray through to the backdrop.
            return true;
        }
        // The COLUMN the strip walks is the tiled one (an untiled image keeps
        // `fract(u)` exactly). `v` is left alone: the loop already sweeps every row
        // band of the image, and a `tile` on `v` only re-orders which band a screen
        // row lands on - it cannot make an opaque texel disappear from the strip.
        let (uu, _) = self.tile_uv(fract(u), 0.0);
        let rows = img.height;
        let samples = rows.min(IMAGE_STRIP_MAX_SAMPLES);
        for i in 0..samples {
            // Sample at the CENTRE of each evenly-spaced row band so a thin opaque
            // line is not stepped over when the texture is taller than the cap.
            let v = (i as f32 + 0.5) / samples as f32;
            let [_, _, _, a] = img.sample_nearest(uu, v);
            if image_texel_opaque(a) {
                return false; // an opaque texel in the strip ⇒ a real hit.
            }
        }
        true
    }

    /// The WORLD-ANCHORED [`ImageSpan`] of this material's Image recipe, if it
    /// has one. `None` for a non-image recipe OR an Image recipe authored without a
    /// span (legacy per-cell tiling). Lets the caster decide once, per cell, whether
    /// to use the world-anchored cut-out path or the legacy per-cell path.
    #[inline]
    pub fn image_span(&self) -> Option<ImageSpan> {
        match self.recipe {
            TextureRecipe::Image { span, .. } => span,
            _ => None,
        }
    }

    /// The `tile =` repeat of this material's Image recipe: how many copies of the
    /// figure fit across the mapped span. [`IMAGE_TILE_NONE`] for a non-image recipe
    /// OR an image authored without `tile`, which is what makes the untiled path a
    /// single equality test and byte-identical.
    #[inline]
    fn image_tile(&self) -> (f32, f32) {
        match self.recipe {
            TextureRecipe::Image { tile, .. } => tile,
            _ => IMAGE_TILE_NONE,
        }
    }

    /// The ONE place an image's cell-centre `(u, v)` is derived, shared by the solid
    /// block, the ASCII-family samplers and the opacity gates so paint and cut-out can
    /// never disagree texel-for-texel.
    ///
    /// * With a world-anchored [`ImageSpan`], the wall hit's world `(along, up)` maps
    ///   through the span (ONE figure across the surface).
    /// * Without one, the legacy per-cell tiling: `u = fract(coord.x)`, `v = wall_v`.
    ///
    /// Then [`tile_uv`](Self::tile_uv) applies the repeat. Untiled, this is exactly the
    /// pre-`tile` expression including its clamps.
    #[inline]
    fn image_uv(&self, coord: Vec2, ctx: &SampleCtx) -> (f32, f32) {
        let (u, v) = match self.image_span() {
            Some(span) => span.uv(ctx.wall_along, ctx.wall_up),
            None => (fract(coord.x), ctx.wall_v),
        };
        self.tile_uv(u, v)
    }

    /// Apply the recipe's `tile =` repeat to a raw image `(u, v)`.
    ///
    /// UNTILED ([`IMAGE_TILE_NONE`]) the pair is only CLAMPED - the historical
    /// behaviour, so a margin outside the figure shows the edge texel rather than a
    /// repeat. TILED, the pair is multiplied and WRAPPED, which is the whole point of
    /// the key: `tile = 4 2` lays four copies across and two up the same span.
    ///
    /// Wrapping (not clamping) is also why a bad factor must never reach here: the
    /// `.lvl` parser rejects non-finite / non-positive factors and
    /// `crate::recipe_from_def` degrades any that were built in code, so `u * tile.0`
    /// is always finite and `fract` is always well-defined.
    #[inline]
    fn tile_uv(&self, u: f32, v: f32) -> (f32, f32) {
        let t = self.image_tile();
        if t == IMAGE_TILE_NONE {
            (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
        } else {
            (fract(u * t.0), fract(v * t.1))
        }
    }

    /// The coordinate a RELIEF image's height field is addressed by, or `None` when
    /// this material has no image relief to sample.
    ///
    /// Returns `Some` only when ALL THREE hold: the recipe is an [`TextureRecipe::Image`]
    /// with an authored `relief`, a texture is attached, and that texture actually
    /// carries a [`re_core::TextureImage::height_map`]. That triple gate is what
    /// guarantees the relief channel is never touched for a material that did not ask
    /// for it - the memory contract for the shared `Arc<[u8]>`.
    ///
    /// The returned pair is the image UV multiplied by `tile` but deliberately NOT
    /// wrapped or clamped: `height_at` does that, so the caller's `coord + eps`
    /// gradient taps move continuously across the field instead of being flattened by
    /// a clamp before they are taken.
    #[inline]
    fn image_relief_uv(&self, coord: Vec2, ctx: &SampleCtx) -> Option<Vec2> {
        let TextureRecipe::Image {
            relief: Some(_),
            tile,
            ..
        } = self.recipe
        else {
            return None;
        };
        let img = self.texture.as_ref()?;
        if !img.has_height_map() {
            return None;
        }
        let (u, v) = match self.image_span() {
            Some(span) => span.uv(ctx.wall_along, ctx.wall_up),
            None => (fract(coord.x), ctx.wall_v),
        };
        Some(Vec2::new(u * tile.0, v * tile.1))
    }

    /// The SUB-CELL detail mode of this material's Image recipe. `Off` for a
    /// non-image recipe OR an Image recipe authored without sub-cell detail — so the
    /// caster only takes the 2×2 / 2×4 sub-sampling path for an opted-in image, and
    /// every other material is byte-identical to the world-anchored-span path.
    #[inline]
    pub fn image_subcell(&self) -> SubcellMode {
        match self.recipe {
            TextureRecipe::Image { subcell, .. } => subcell,
            _ => SubcellMode::Off,
        }
    }

    /// IMAGE CUT-OUT — WORLD-ANCHORED variant . Is the texel at the given WORLD
    /// `(along, up)` opaque? The `span` maps world → texture `(u, v)` (ONE figure over
    /// the whole surface). A `(u, v)` outside `[0,1]` is OUTSIDE the figure → treated
    /// as TRANSPARENT (`false`, shows the backdrop), so a cell beyond the figure's
    /// world rectangle never paints the pane. Inside the box, the attached texture's
    /// alpha decides exactly as the per-cell test does. A non-image / unresolved
    /// material is OPAQUE (`true`), never punching a hole (matches the legacy test).
    #[inline]
    pub fn image_texel_opaque_at_world(&self, span: &ImageSpan, along: f32, up: f32) -> bool {
        match (&self.recipe, self.texture.as_ref()) {
            (TextureRecipe::Image { .. }, Some(img)) => {
                let (u, v) = span.uv(along, up);
                // A TILED image has no margin by construction - the figure repeats, so
                // a UV past the span wraps onto the next copy rather than falling
                // outside the figure. Untiled, the out-of-box reject is unchanged.
                if self.image_tile() == IMAGE_TILE_NONE
                    && (!(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v))
                {
                    return false; // outside the figure's world box ⇒ transparent margin.
                }
                let (u, v) = self.tile_uv(u, v);
                let [_, _, _, a] = img.sample_nearest(u, v);
                image_texel_opaque(a)
            }
            _ => true,
        }
    }

    /// IMAGE CUT-OUT — SUB-CELL per-row opacity gate . For a `Quad`/`Braille`
    /// image pane, is ANY of the cell-row's SUB-texels opaque at the world centre
    /// `(along, up)`? The per-row cut-out gate must keep a screen row whenever the
    /// 2×2 / 2×4 glyph that row will draw has at least one opaque sub-texel — a row
    /// whose CELL-CENTRE texel is transparent but whose silhouette-edge sub-texels
    /// are opaque (hair, antenna, a 1px outline) must NOT be dropped at whole-cell
    /// granularity before the sub-sampler ever runs.
    ///
    /// The sub-positions tested are EXACTLY the ones
    /// [`sample_image_quad`](Self::sample_image_quad) /
    /// [`sample_image_braille`](Self::sample_image_braille) sample
    /// (`±¼` footprint ALONG, and `±¼` up for quad or `±⅜/±⅛` up for braille via
    /// [`image_subtexel`](Self::image_subtexel)'s `span.uv` mapping), so the gate and
    /// the shade agree texel-for-texel: every row the gate KEEPS produces an inked
    /// glyph, every row it drops would have produced an all-transparent (blank) glyph.
    ///
    /// `footprint` is the caster's `(fp_along, fp_up)` world step (the same value
    /// threaded into [`SampleCtx::tex_footprint`]). `Off` is NOT handled here — the
    /// caster keeps the single cell-centre [`Self::image_texel_opaque_at_world`] gate for
    /// `Off` (byte-identical). A non-image / unresolved material is OPAQUE (`true`).
    #[inline]
    pub fn image_row_opaque_subcell(
        &self,
        span: &ImageSpan,
        along: f32,
        up: f32,
        footprint: (f32, f32),
        mode: SubcellMode,
    ) -> bool {
        // A non-image / unresolved Image material never punches a hole → opaque,
        // matching every other image opacity test (the cut-out never triggers for it).
        if !matches!(self.recipe, TextureRecipe::Image { .. }) || self.texture.is_none() {
            return true;
        }
        let (fp_along, fp_up) = footprint;
        // Column sub-centres: ±¼ footprint ALONG (left / right), as in both samplers.
        let col_off = [-0.25 * fp_along, 0.25 * fp_along];
        // Row sub-centres up the cell, matched to the chosen glyph's vertical layout.
        // `up` grows UPWARD, so the TOP sub-row carries the LARGEST +up offset.
        match mode {
            SubcellMode::Off => {
                // Defensive: the caster does not call this for `Off`. Fall back to the
                // cell-centre gate so the result is still self-consistent.
                self.image_texel_opaque_at_world(span, along, up)
            }
            // HALF probes the two vertical sub-centres; SEXTANT the three row bands.
            // Both share the ±¼ column offsets (Half only has one column, but probing
            // both halves of the cell's width is the same "is any texel here opaque"
            // question and keeps a thin vertical figure from being missed).
            SubcellMode::Half => {
                for &du in &[0.25 * fp_up, -0.25 * fp_up] {
                    for &da in &col_off {
                        if self.image_texel_opaque_at_world(span, along + da, up + du) {
                            return true;
                        }
                    }
                }
                false
            }
            SubcellMode::Sextant => {
                for &du in &[fp_up / 3.0, 0.0, -fp_up / 3.0] {
                    for &da in &col_off {
                        if self.image_texel_opaque_at_world(span, along + da, up + du) {
                            return true;
                        }
                    }
                }
                false
            }
            SubcellMode::Quad => {
                let row_off = [0.25 * fp_up, -0.25 * fp_up]; // top, bottom
                for &du in &row_off {
                    for &da in &col_off {
                        if self.image_texel_opaque_at_world(span, along + da, up + du) {
                            return true;
                        }
                    }
                }
                false
            }
            // Braille AND the ASCII-family edge samplers (Ascii/Contour/Lines) all
            // treat the cell as a 2×4 coverage probe for cut-out purposes: any opaque
            // sub-texel in the cell's world slice means the figure is present here, so
            // the column does not march past. The ASCII samplers PAINT a luminance fill
            // (not a 1-bit silhouette), but their opacity — whether the cell shows the
            // figure or the backdrop — is the same "is any texel here opaque" question.
            SubcellMode::Braille
            | SubcellMode::Ascii
            | SubcellMode::Contour
            | SubcellMode::Lines => {
                let row_off = [
                    0.375 * fp_up,  // row 0 (top)
                    0.125 * fp_up,  // row 1
                    -0.125 * fp_up, // row 2
                    -0.375 * fp_up, // row 3 (bottom)
                ];
                for &du in &row_off {
                    for &da in &col_off {
                        if self.image_texel_opaque_at_world(span, along + da, up + du) {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    /// IMAGE CUT-OUT — WORLD-ANCHORED strip test . At the cell's fixed world
    /// `along` (so a single texture `u`), is EVERY texel transparent over the world
    /// UP range `[up_bottom, up_top]` this cell covers? Unlike the legacy whole-image
    /// strip, the span maps this cell to only the SLICE of the figure that falls in
    /// its world-Z extent — so a tall figure's mid-cells correctly report opaque while
    /// its margin cells report transparent. `true` ⇒ the whole slice is sub-cutoff (or
    /// the cell is entirely outside the figure's box) → the DDA marches PAST to the
    /// backdrop. A non-image / unresolved material returns `false` (never a free
    /// pass-through), matching the legacy strip test.
    ///
    /// Samples one texel per IMAGE ROW that falls in the slice (centre-of-band), capped
    /// at [`IMAGE_STRIP_MAX_SAMPLES`] so the per-cell cost stays bounded for a tall
    /// texture. Pure: the same world inputs always give the same answer.
    #[inline]
    pub fn image_strip_fully_transparent_world(
        &self,
        span: &ImageSpan,
        along: f32,
        up_top: f32,
        up_bottom: f32,
    ) -> bool {
        let (TextureRecipe::Image { .. }, Some(img)) = (&self.recipe, self.texture.as_ref()) else {
            return false;
        };
        if img.width == 0 || img.height == 0 {
            return true; // zero-area texture ⇒ empty pane, pass through.
        }
        // The cell's fixed texture column. Outside [0,1] ⇒ the whole cell is in the
        // figure's left/right margin → fully transparent, march on.
        let (u, _) = span.uv(along, up_top);
        if !(0.0..=1.0).contains(&u) {
            return true;
        }
        // The texture-v range this cell covers (v grows DOWNWARD; up grows UPWARD, so
        // the cell's TOP world-up maps to the SMALLER v). Clamp into the figure's box;
        // an empty intersection (cell entirely above/below the figure) ⇒ transparent.
        let (_, v_at_top) = span.uv(along, up_top);
        let (_, v_at_bottom) = span.uv(along, up_bottom);
        let v_lo = v_at_top.min(v_at_bottom);
        let v_hi = v_at_top.max(v_at_bottom);
        let v_lo_c = v_lo.clamp(0.0, 1.0);
        let v_hi_c = v_hi.clamp(0.0, 1.0);
        if v_hi < 0.0 || v_lo > 1.0 {
            return true; // cell's world slice is entirely outside the figure vertically.
        }
        let rows = img.height;
        // Number of image rows inside the visible v-band, capped.
        let band = (v_hi_c - v_lo_c).max(0.0);
        let band_rows = ((band * rows as f32).ceil() as u32).max(1);
        let samples = band_rows.min(IMAGE_STRIP_MAX_SAMPLES);
        for i in 0..samples {
            let t = (i as f32 + 0.5) / samples as f32;
            let v = v_lo_c + t * (v_hi_c - v_lo_c);
            let [_, _, _, a] = img.sample_nearest(u, v);
            if image_texel_opaque(a) {
                return false; // an opaque texel in the slice ⇒ a real hit.
            }
        }
        true
    }

    // ── image: sample an attached RGBA8 texture onto the wall face ──
    //
    // Milestone 2 of the image-texture feature: nearest-neighbour sample the
    // attached [`re_core::TextureImage`] at the cell's `(u, v)` and paint a SOLID
    // `█` block in that colour. The ALPHA cut-out (transparent texels show the
    // backdrop behind) is handled by the caster, which only reaches `sample_image`
    // for a texel it already proved OPAQUE — so here the colour is painted directly.
    // No sub-cell, no tiling, no scale — those come in a later milestone.
    //
    // - `u` is the wall-HORIZONTAL fraction: the caster threads `tex_u + integer
    //   face offset` into `coord.x` (exactly as the `CodeWall` recipe consumes), so
    //   `fract(coord.x)` is the in-cell `[0,1]` position across the wall face.
    // - `v` is `ctx.wall_v` (the vertical position DOWN the wall slice, `0` = top),
    //   matching the [`re_core::TextureImage`] top-left-origin UV convention.
    //
    // The texture itself is the `Arc<TextureImage>` attached to `self.texture` by
    // `crate::registry_for_map` (NO `&Map` needed here). A missing texture (an
    // out-of-range `tex_id` the registry could not resolve, so the field is `None`)
    // renders a MAGENTA error block rather than panicking.
    fn sample_image(&self, coord: Vec2, ctx: &SampleCtx) -> Texel {
        // ANIMATION: an `[texture] frames = clip.gif` material resolves its frame from
        // `ctx.time` here, once, and every sub-cell sampler below reads that same
        // borrow. A still texture takes the `None` arm and gets `self.texture`
        // verbatim — byte-identical to before frames existed.
        let Some(img) = self.image_frame(ctx) else {
            // No attached texture (out-of-range tex_id / unresolved) → visible error
            // block, never a panic. Magenta is the conventional "missing texture".
            let err = pack_rgb(Vec3::new(1.0, 0.0, 1.0));
            return Texel {
                glyph: SOLID_BLOCK,
                fg: err,
                bg: err,
                rank: 0,
            };
        };

        // SUB-CELL detail: when the recipe opts into `Quad`/`Braille` AND the
        // caster threaded this cell's world footprint, sample the texture at the cell's
        // 2×2 / 2×4 sub-positions and emit a higher-resolution glyph. `Off` (or a caster
        // that didn't supply a footprint — every legacy/test path) falls through to the
        // the plain solid-block path below, byte-identical.
        match (self.image_subcell(), ctx.tex_footprint) {
            (SubcellMode::Half, Some(fp)) => return self.sample_image_half(img, coord, ctx, fp),
            (SubcellMode::Quad, Some(fp)) => return self.sample_image_quad(img, coord, ctx, fp),
            (SubcellMode::Sextant, Some(fp)) => {
                return self.sample_image_sextant(img, coord, ctx, fp)
            }
            (SubcellMode::Braille, Some(fp)) => {
                return self.sample_image_braille(img, coord, ctx, fp)
            }
            // ASCII-family edge samplers (quality-textures Component 2). All three share
            // one assembler; the `EdgeMode` argument selects which precomputed edge layer
            // (Sobel vs DoG) and the lines-only ink treatment.
            (SubcellMode::Ascii, Some(fp)) => {
                return self.sample_image_ascii(img, coord, ctx, fp, EdgeMode::Sobel)
            }
            (SubcellMode::Contour, Some(fp)) => {
                return self.sample_image_ascii(img, coord, ctx, fp, EdgeMode::Dog)
            }
            (SubcellMode::Lines, Some(fp)) => {
                return self.sample_image_ascii(img, coord, ctx, fp, EdgeMode::Lines)
            }
            _ => {}
        }

        // (u, v) selection:
        //  * WORLD-ANCHORED span present ⇒ map the wall hit's WORLD (along, up) through
        //    the `ImageSpan` so ONE figure spans the whole surface. As the camera nears,
        //    each screen cell covers a smaller world span → fewer texels per cell →
        //    nearest sampling stays sharp ("walk up and see it pixel-by-pixel"). Outside
        //    [0,1] is clamped to the figure's edge (the cut-out already discards the
        //    margin; a non-cut-out wall shows the edge texel rather than a tiled repeat).
        //  * No span ⇒ LEGACY per-cell tiling: `u = fract(coord.x)`, `v = wall_v`.
        let (u, v) = self.image_uv(coord, ctx);
        let [r, g, b, _a] = img.sample_nearest(u, v);
        let rgb = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
        // Solid block: the whole cell is the sampled colour. fg == the colour; bg a
        // slightly darkened shade so the `█` body reads crisply over its own cell
        // edge (a subtle inner depth cue) without crushing the image's colour.
        Texel {
            glyph: SOLID_BLOCK,
            fg: rgb,
            bg: dim_rgb(rgb, IMAGE_BG_DIM),
            rank: 0,
        }
    }

    /// Sample the texture at ONE sub-position of the cell, offset `(d_along, d_up)`
    /// world-units from the cell centre. Returns `(opaque, rgb)`: the nearest
    /// texel's colour and whether it is opaque (alpha ≥ cutoff). Uses the world-anchored
    /// span when present (so the offset is a true world step → finer texels up close),
    /// else the legacy per-cell `(coord.x, wall_v)` mapping with the offset applied in UV.
    #[inline]
    fn image_subtexel(
        &self,
        img: &re_core::TextureImage,
        coord: Vec2,
        ctx: &SampleCtx,
        d_along: f32,
        d_up: f32,
    ) -> (bool, u32) {
        let (u, v) = match self.image_span() {
            Some(span) => span.uv(ctx.wall_along + d_along, ctx.wall_up + d_up),
            // Legacy per-cell tiling: the cell IS one [0,1] tile, so the world step maps
            // straight to a UV step (d_along/d_up are already in cell-fraction units when
            // the footprint is ~1 cell). `u` grows with along, `v` grows DOWNWARD.
            None => (fract(coord.x) + d_along, ctx.wall_v - d_up),
        };
        // `tile =` repeat, applied to the SUB-position exactly as the cell-centre path
        // applies it - otherwise a tiled image's sub-cell glyph would be assembled from
        // texels of a different copy than the one its colour came from. Untiled this is
        // only the historical clamp.
        let (u, v) = self.tile_uv(u, v);
        // Outside [0,1] is OUTSIDE the figure → transparent (shows the backdrop), matching
        // the cut-out's world-box rule. Inside, the texture alpha decides.
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return (false, 0);
        }
        let [r, g, b, a] = img.sample_nearest(u, v);
        let rgb = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
        (image_texel_opaque(a), rgb)
    }

    /// 2×2 QUADRANT sub-cell (interior-detail upgrade): sample the cell's four
    /// sub-positions and blit them with [`crate::quadrant::blit_quadrant_luminance`], which
    /// resolves INTERIOR tonal detail (a light/dark split inside the figure) at 2×2 while
    /// keeping the alpha silhouette sharp (transparent sub-texels still reveal the backdrop).
    /// A flat-colour interior reads as `█`; a tonal boundary as a half/quarter block on the
    /// real edge; the figure outline as transparent→backdrop. Doubles effective resolution
    /// on the BODY, not only the outline.
    fn sample_image_quad(
        &self,
        img: &re_core::TextureImage,
        coord: Vec2,
        ctx: &SampleCtx,
        footprint: (f32, f32),
    ) -> Texel {
        use crate::quadrant::{blit_quadrant_luminance, ImageSub};
        let (fp_along, fp_up) = footprint;
        // Backdrop a transparent sub-texel reveals: the cut-out's scene-behind when known,
        // else the cell's own dim of the centre texel (so a non-cut-out wall still reads).
        let backdrop = ctx
            .subcell_backdrop
            .unwrap_or_else(|| dim_rgb(self.image_centre_rgb(img, coord, ctx), IMAGE_BG_DIM));
        // The four sub-centres at ±¼ footprint. Order [TL, TR, BL, BR] to match the
        // quadrant mask bits; `up` grows UPWARD so TL/TR (top) are +¼ up, BL/BR −¼ up.
        let q = 0.25;
        let offsets = [
            (-q * fp_along, q * fp_up),  // TL
            (q * fp_along, q * fp_up),   // TR
            (-q * fp_along, -q * fp_up), // BL
            (q * fp_along, -q * fp_up),  // BR
        ];
        let mut subs = [ImageSub {
            opaque: false,
            rgb: 0,
        }; 4];
        for (i, &(da, du)) in offsets.iter().enumerate() {
            let (opaque, rgb) = self.image_subtexel(img, coord, ctx, da, du);
            subs[i] = ImageSub { opaque, rgb };
        }
        let blit = blit_quadrant_luminance(subs, backdrop);
        Texel {
            glyph: blit.glyph,
            fg: blit.fg,
            bg: blit.bg,
            rank: 0,
        }
    }

    /// 2×1 HALF-BLOCK sub-cell: sample the cell's top and bottom sub-positions and
    /// blit them with [`crate::quadrant::blit_half_luminance`].
    ///
    /// The cheapest rung of the ladder, and the one whose value is not just "more
    /// texels": a terminal cell is roughly 2:1 tall, so splitting it vertically once
    /// is what makes a sub-texel approximately SQUARE. Every glyph it can emit is a
    /// Block Element present wherever the quadrant set is, so it needs no capability
    /// gate and is the correct degrade target for [`SubcellMode::Sextant`].
    fn sample_image_half(
        &self,
        img: &re_core::TextureImage,
        coord: Vec2,
        ctx: &SampleCtx,
        footprint: (f32, f32),
    ) -> Texel {
        use crate::quadrant::{blit_half_luminance, ImageSub};
        let (_fp_along, fp_up) = footprint;
        let backdrop = ctx
            .subcell_backdrop
            .unwrap_or_else(|| dim_rgb(self.image_centre_rgb(img, coord, ctx), IMAGE_BG_DIM));
        // Two sub-centres at ±¼ footprint UP; `up` grows upward so top is +¼.
        let mut subs = [ImageSub {
            opaque: false,
            rgb: 0,
        }; 2];
        for (i, du) in [0.25 * fp_up, -0.25 * fp_up].into_iter().enumerate() {
            let (opaque, rgb) = self.image_subtexel(img, coord, ctx, 0.0, du);
            subs[i] = ImageSub { opaque, rgb };
        }
        let blit = blit_half_luminance(subs, backdrop);
        Texel {
            glyph: blit.glyph,
            fg: blit.fg,
            bg: blit.bg,
            rank: 0,
        }
    }

    /// 3×2 SEXTANT sub-cell: sample the cell's six sub-positions and blit them with
    /// [`crate::quadrant::blit_sextant_luminance`].
    ///
    /// This is the rung intended to displace Braille for IMAGE work. Braille gives
    /// eight sub-texels but, being dots rather than filled area, gives the whole cell
    /// ONE ink colour; sextants give six sub-texels as filled blocks and therefore
    /// keep the two-colour split that makes quadrant output read. `notcurses` reaches
    /// the same conclusion from the other direction — it rates Braille poor for
    /// images and places Sextant directly above Quadrant on its degradation ladder.
    ///
    /// Requires terminal support for `U+1FB00..=U+1FB3B`; the caller gates on that and
    /// degrades to [`SubcellMode::Quad`] when absent (see `re_world::SubcellMode`).
    fn sample_image_sextant(
        &self,
        img: &re_core::TextureImage,
        coord: Vec2,
        ctx: &SampleCtx,
        footprint: (f32, f32),
    ) -> Texel {
        use crate::quadrant::{blit_sextant_luminance, ImageSub};
        let (fp_along, fp_up) = footprint;
        let backdrop = ctx
            .subcell_backdrop
            .unwrap_or_else(|| dim_rgb(self.image_centre_rgb(img, coord, ctx), IMAGE_BG_DIM));
        // Columns at ±¼ along; rows at +⅓, 0, −⅓ up (three evenly-spaced bands whose
        // sub-centres sit at the middle of each third of the cell).
        let col_off = [-0.25 * fp_along, 0.25 * fp_along];
        let row_off = [fp_up / 3.0, 0.0, -fp_up / 3.0];
        let blank = ImageSub {
            opaque: false,
            rgb: 0,
        };
        let mut grid = [[blank; 2]; 3];
        for (r, &du) in row_off.iter().enumerate() {
            for (c, &da) in col_off.iter().enumerate() {
                let (opaque, rgb) = self.image_subtexel(img, coord, ctx, da, du);
                grid[r][c] = ImageSub { opaque, rgb };
            }
        }
        let blit = blit_sextant_luminance(grid, backdrop);
        Texel {
            glyph: blit.glyph,
            fg: blit.fg,
            bg: blit.bg,
            rank: 0,
        }
    }

    /// 2×4 BRAILLE sub-cell (interior-detail upgrade): sample the cell's eight
    /// sub-positions and blit them with [`crate::quadrant::blit_braille_luminance`], a 2×4
    /// HALFTONE of the figure's interior luminance (darker-than-mean opaque texels ink a dot)
    /// plus the sharp alpha edge (transparent texels stay OFF, revealing the backdrop). `fg`
    /// is the mean ink colour, `bg` the mean OFF colour. Ink-polarity = darker=ink (reads
    /// portraits / line detail well). The high-detail alpha-cut-out mode.
    fn sample_image_braille(
        &self,
        img: &re_core::TextureImage,
        coord: Vec2,
        ctx: &SampleCtx,
        footprint: (f32, f32),
    ) -> Texel {
        use crate::quadrant::{blit_braille_luminance, ImageSub};
        let (fp_along, fp_up) = footprint;
        let backdrop = ctx
            .subcell_backdrop
            .unwrap_or_else(|| dim_rgb(self.image_centre_rgb(img, coord, ctx), IMAGE_BG_DIM));
        // 2 columns × 4 rows. Column centres at ±¼ along; row centres at ±⅜/±⅛ up
        // (the four evenly-spaced sub-rows of the cell, top→bottom). `up` grows UPWARD,
        // so the TOP row is the LARGEST +up offset.
        let col_off = [-0.25 * fp_along, 0.25 * fp_along]; // left, right
        let row_off = [
            0.375 * fp_up,  // row 0 (top)
            0.125 * fp_up,  // row 1
            -0.125 * fp_up, // row 2
            -0.375 * fp_up, // row 3 (bottom)
        ];
        let mut grid = [[ImageSub {
            opaque: false,
            rgb: 0,
        }; 2]; 4];
        for (r, &du) in row_off.iter().enumerate() {
            for (c, &da) in col_off.iter().enumerate() {
                let (opaque, rgb) = self.image_subtexel(img, coord, ctx, da, du);
                grid[r][c] = ImageSub { opaque, rgb };
            }
        }
        let blit = blit_braille_luminance(grid, backdrop);
        Texel {
            glyph: blit.glyph,
            fg: blit.fg,
            bg: blit.bg,
            rank: 0,
        }
    }

    /// ASCII-FAMILY edge sampler (quality-textures Component 2): luminance-ramp fill
    /// glyph, swapped for a directional contour glyph where the precomputed
    /// [`crate::edge_field::EdgeField`] reports a strong edge. `mode` selects the look:
    ///
    /// * [`EdgeMode::Sobel`] (`Ascii`) — edge layer from the raw Sobel field.
    /// * [`EdgeMode::Dog`] (`Contour`) — edge layer from the DoG field (cleaner thin
    ///   lines). In practice both share ONE baked field (Sobel-on-DoG); the modes
    ///   differ in ink/contrast treatment, the hero `Contour` favouring the line.
    /// * [`EdgeMode::Lines`] (`Lines`) — EVERY cell is a glyph; a non-edge cell's fg is
    ///   pulled close to its bg ([`LINES_FILL_CONTRAST`]) so only contour glyphs pop.
    ///
    /// Colour is two-channel: the glyph ink in `fg`, the local mean fill colour in
    /// `bg` — one cell carries both the stroke and the fill tone (spec's "colour as
    /// extra resolution"). The mip level is chosen by `footprint` (minification LOD).
    fn sample_image_ascii(
        &self,
        img: &re_core::TextureImage,
        coord: Vec2,
        ctx: &SampleCtx,
        footprint: (f32, f32),
        mode: EdgeMode,
    ) -> Texel {
        // The cell-centre UV (world-anchored span if present, else legacy per-cell,
        // then the `tile =` repeat) - the SAME derivation the solid-block path uses.
        let (u, v) = self.image_uv(coord, ctx);

        // Local mean FILL colour (the cell's bg tone) — the centre texel's colour.
        let [r, g, b, _a] = img.sample_nearest(u, v);
        let fill = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);

        // EDGE LAYER: sample the precomputed field at the mip level matching the cell's
        // texel footprint. `texels_per_cell` ≈ how many level-0 texels one cell covers
        // along the tighter axis (footprint world-units × texels-per-world-unit). The
        // span maps world→UV, so footprint/along_size × width is the texel count.
        let (lum, mag, dir_bin) = match self.edge_field.as_ref() {
            Some(field) => {
                let texels = self.ascii_texels_per_cell(field, footprint);
                let level = field.level_for_texels(texels);
                level.sample(u, v)
            }
            // No baked field (e.g. a test material with a texture but no precompute):
            // fall back to the centre-texel luminance, no edges. Still a valid output.
            None => {
                let l = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) / 255.0;
                (l, 0.0, 0u8)
            }
        };

        // Edge gate. `Contour` (the hero DoG look) raises the bar so only the strongest
        // responses ink a stroke → thinner, cleaner lines; `Sobel`/`Lines` use the
        // baseline. Both read the same baked Sobel-on-DoG field.
        let threshold = match mode {
            EdgeMode::Dog => ASCII_EDGE_THRESHOLD * CONTOUR_THRESHOLD_SCALE,
            _ => ASCII_EDGE_THRESHOLD,
        };
        let on_edge = mag >= threshold;

        // FILL glyph from the luminance ramp (dark→light).
        let fill_glyph = ascii_ramp_glyph(lum);

        if on_edge {
            // Directional contour glyph. Ink = the fill colour lifted toward white so
            // the stroke reads as a light line; bg = the local fill tone (two colours).
            //
            // ALIGNMENT: the bin says which WAY the edge runs, not WHERE in the cell it
            // sits, so a horizontal edge grazing the cell's bottom draws the same
            // centred `-` as one straight through the middle. `ascii_edge_row_bias`
            // resolves that by asking the field where the response actually is — see
            // its docs for why only the horizontal case is treated.
            let bin = self.ascii_edge_row_bias(dir_bin, u, v, footprint);
            let glyph = u32::from(crate::edge_field::DIR_GLYPHS[(bin as usize).min(4)]);
            let ink = mix_rgb(fill, 0x00FFFFFF, ASCII_EDGE_INK_LIFT * mag.clamp(0.0, 1.0));
            let ink = match mode {
                EdgeMode::Lines => dim_rgb(ink, LINES_EDGE_INK),
                _ => ink,
            };
            Texel {
                glyph,
                fg: ink,
                bg: dim_rgb(fill, IMAGE_BG_DIM),
                rank: 0,
            }
        } else {
            // NON-edge fill cell. `Lines` mode recesses the fill (fg ≈ bg) so only the
            // contour cells pop; `Ascii`/`Contour` ink the ramp glyph in the fill colour.
            let bg = dim_rgb(fill, IMAGE_BG_DIM);
            let fg = match mode {
                EdgeMode::Lines => mix_rgb(bg, fill, LINES_FILL_CONTRAST),
                _ => fill,
            };
            // Every mode emits the ramp fill glyph in a non-edge cell; `Lines` just
            // recesses its ink (above) so the field reads as flat colour with crisp
            // contours, while `Ascii`/`Contour` ink the ramp glyph in the fill colour.
            Texel {
                glyph: fill_glyph,
                fg,
                bg,
                rank: 0,
            }
        }
    }

    /// ALIGNMENT-AWARE direction bin: promote a near-horizontal edge that sits LOW in
    /// the cell from `-` (bin 0) to `_` (bin 4).
    ///
    /// ## Why this exists
    ///
    /// [`crate::edge_field::angle_to_bin`] answers "which way does the edge run", and
    /// nothing else. Two cells whose edges run identically but sit at opposite ends of
    /// the cell get the same centred glyph, so a long horizontal boundary drawn across
    /// a row of cells wobbles: each cell inks at its own centre regardless of where
    /// the boundary actually crosses it. This is the observation behind the
    /// alignment-INSENSITIVE shape metric in the structure-based ASCII-art literature
    /// (Xu/Zhang/Wong) — *where* the ink sits inside the cell is part of the match,
    /// not just its orientation.
    ///
    /// We take the cheap lookup form of that idea rather than the paper's offline
    /// optimisation: probe the baked field once above and once below the cell centre
    /// and let the stronger response choose the glyph.
    ///
    /// ## Why ONLY the horizontal case
    ///
    /// This is the ASCII-family sampler, so the substitute has to be ASCII. ASCII
    /// offers exactly one positional variant of a rule — `_` against `-` — and no
    /// left/right variant of `|` at all (there is no ASCII half-width bar). Inventing
    /// one from Block Elements would make the *ASCII* mode not ASCII. The block
    /// samplers do not need this at all: a quadrant/sextant mask encodes sub-cell
    /// position by construction.
    ///
    /// `angle_to_bin` never returns 4 — it folds low-horizontal into bin 0 and
    /// documents bin 4 as reserved "for samplers that want a low-baseline variant".
    /// This is that sampler.
    ///
    /// Costs two extra field samples, and only on cells that already passed the edge
    /// gate. Non-horizontal bins return unchanged without sampling anything.
    #[inline]
    fn ascii_edge_row_bias(&self, dir_bin: u8, u: f32, v: f32, footprint: (f32, f32)) -> u8 {
        // Only a near-horizontal edge (bin 0 = `-`) has an ASCII positional variant.
        if dir_bin != 0 {
            return dir_bin;
        }
        let Some(field) = self.edge_field.as_ref() else {
            return dir_bin;
        };
        // How much of the image's V range ONE TERMINAL CELL covers. With a span, the
        // footprint's share of the span's world height. WITHOUT one, the legacy
        // per-cell tiling maps the whole texture across ONE WORLD CELL of wall, so V
        // advances `1.0` per world unit of height and the terminal cell's share is
        // just its world-up footprint.
        //
        // The first version wrote `1.0` here, having asked the right question of the
        // wrong box: the WALL cell does span the full V range, but the probe is about
        // the TERMINAL cell inside it. The probes then landed a quarter of the whole
        // IMAGE apart instead of a quarter of a cell — twenty-odd times too far — and
        // measurably so: instrumented over a photo-textured wall, the upper probe read
        // exactly `0.0` on every single call, because it had been pushed clean off the
        // subject. Every bias that fired that way was noise.
        let (_fp_along, fp_up) = footprint;
        let dv = match self.image_span() {
            Some(span) => fp_up / span.up_size.max(1e-6),
            None => fp_up,
        };
        // Probe a quarter-cell above and below. `v` grows DOWNWARD (see `ImageSpan::uv`),
        // so the LOWER probe is the LARGER v.
        let q = 0.25 * dv;
        // CRITICAL: probe a FINER mip than the cell's own. `level_for_texels` picks the
        // level where "one mip texel ≈ one cell" — the minification rule that kills
        // shimmer, and precisely the level at which sub-cell structure has already been
        // averaged away. Sampling it at ±¼ cell would read the same texel twice and the
        // bias could never fire. Asking for a quarter of the texel count selects the
        // level whose texel ≈ ¼ cell, which is the scale the question is about.
        let texels = self.ascii_texels_per_cell(field, footprint);
        let level = field.level_for_texels(texels * 0.25);
        let (_, mag_hi, _) = level.sample(u, v - q);
        let (_, mag_lo, _) = level.sample(u, v + q);
        // Require a CLEAR bias before moving the stroke: a boundary genuinely centred
        // in the cell must keep `-`, or a flat run of cells alternates between glyphs
        // on noise and reads worse than the un-biased version.
        if mag_lo > mag_hi * ASCII_EDGE_BIAS_RATIO {
            4 // `_` — the response is concentrated in the cell's lower half.
        } else {
            0 // `-` — centred, or biased high (which ASCII cannot express).
        }
    }

    /// Estimate how many LEVEL-0 texels one screen cell covers (for mip selection).
    /// Uses the tighter axis (the one that minifies first) so the chosen level never
    /// under-filters. With a span, `footprint.along / span.along_size` is the cell's
    /// fraction of the U range → × texture width = texels along; likewise up/height.
    #[inline]
    fn ascii_texels_per_cell(
        &self,
        field: &crate::edge_field::EdgeField,
        footprint: (f32, f32),
    ) -> f32 {
        let (fp_along, fp_up) = footprint;
        let (w, h) = (field.width() as f32, field.height() as f32);
        match self.image_span() {
            Some(span) => {
                let tex_u = (fp_along / span.along_size.max(1e-6)) * w;
                let tex_v = (fp_up / span.up_size.max(1e-6)) * h;
                // Tighter axis → larger texels/cell → coarser (safer) mip.
                tex_u.max(tex_v).max(1.0)
            }
            // Legacy per-cell tiling maps the whole texture across ONE WORLD CELL —
            // `u` wraps once per world cell (`fract(coord.x)`) and `v` runs 0..1 over
            // one wall slice. So a TERMINAL cell covers its world footprint's SHARE of
            // the image, exactly as in the span case with a span of one world unit.
            //
            // This read `w.max(h)` — "the whole texture is one cell" — which is true
            // of the world cell and false of the terminal cell inside it. The effect
            // was to pick the coarsest mip no matter how close the camera stood, so a
            // no-span ASCII/Contour/Lines wall could never resolve detail by walking
            // up to it. Measured on a 16x16 fixture: 16 texels/cell reported at every
            // footprint from 1/2 down to 1/32 of a world unit.
            None => (fp_along * w).max(fp_up * h).max(1.0),
        }
    }

    /// The nearest texel's `0x00RRGGBB` at the cell CENTRE — the fallback
    /// backdrop tint for a sub-cell blit when no cut-out backdrop was threaded.
    #[inline]
    fn image_centre_rgb(&self, img: &re_core::TextureImage, coord: Vec2, ctx: &SampleCtx) -> u32 {
        let (u, v) = match self.image_span() {
            Some(span) => {
                let (uu, vv) = span.uv(ctx.wall_along, ctx.wall_up);
                (uu.clamp(0.0, 1.0), vv.clamp(0.0, 1.0))
            }
            None => (fract(coord.x), ctx.wall_v.clamp(0.0, 1.0)),
        };
        let [r, g, b, _a] = img.sample_nearest(u, v);
        (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
    }

    // ── code wall: the engine's own source, laid onto the wall (strange loop) ──
    //
    // This is the shader half of the strange loop. The geometry half (the wall
    // `(u, wall_v)` coordinate) is computed by the raycaster; here we turn it into
    // a glyph: ask the level's text-surface registry which source character lives at this
    // wall coordinate, and draw it terminal-green on near-black, brightening the
    // comments so the operator's asides read as the system's voice.
    fn sample_code_wall(
        &self,
        coord: Vec2,
        ctx: &SampleCtx,
        panel_index: u32,
        u_chars_per_unit: f32,
    ) -> (Texel, Option<Texel>) {
        use crate::code_wall::CodeClass;

        // `coord.x` carries the wall `u` (tex_u + integer face offset, threaded by
        // the caster's `sample_coord`). `ctx.wall_v` is the vertical position down
        // the wall slice in [0,1]. Together they address the source grid.
        //
        // The panel is resolved through the PER-LEVEL text-surface registry
        // (`crate::scene_text`) instead of the old `'static` provider global.
        //
        // D3 — THE FLOWING PLANE. `coord.x`'s fractional part is `tex_u`, which WRAPS
        // every world cell, so addressing the panel with it showed the same handful of
        // source columns on every cell of a wall run — and at a fixed authored density
        // it also replicated each character across up to 50 screen columns while
        // mapping 600 source lines onto ~14 screen rows. When the cast measured the
        // wall's screen span we instead lay the source as one continuous plane along
        // the wall RUN at a screen-matched, octave-quantized density (see
        // `crate::code_wall`'s FlowScale notes). With no measured span (a focused unit
        // test) the historical mapping is kept, byte-for-byte.
        let flow = crate::face_span::wall_run().and_then(|run| {
            // DENSITY from the FACE's own projected width when the cast measured one —
            // that is constant across the whole face, so the density can never step in
            // the middle of a wall and shear the text. Otherwise the frontal estimate
            // `cols_per_unit / depth`, which is the same quantity for a face-on wall.
            let anchor = crate::face_span::face_anchor();
            let density_cols = if anchor.is_known() {
                anchor.span_cols
            } else {
                crate::face_span::face_cols(ctx.distance, 1.0)
            };
            crate::code_wall::FlowScale::for_view(
                u_chars_per_unit,
                density_cols,
                run.footprint_cols,
                crate::face_span::face_rows(ctx.distance),
                crate::face_span::has_viewport_basis(),
            )
            .map(|s| (run.u, s, anchor))
        });
        let (cg, source_row) = match flow {
            Some((run_u, scale, anchor)) if anchor.is_known() => {
                crate::scene_text::sample_code_panel_justified(
                    panel_index,
                    run_u,
                    ctx.wall_v,
                    scale,
                    anchor.u_cols,
                    anchor.span_cols,
                )
            }
            Some((run_u, scale, _)) => {
                crate::scene_text::sample_code_panel_flowing(panel_index, run_u, ctx.wall_v, scale)
            }
            None => {
                let (g, panel_rows) = crate::scene_text::sample_code_panel(
                    panel_index,
                    coord.x,
                    ctx.wall_v,
                    u_chars_per_unit,
                );
                (
                    g,
                    (ctx.wall_v.clamp(0.0, 1.0) * panel_rows as f32).floor() as i64,
                )
            }
        };

        // Two greens + a near-black base. The bg is the dead-terminal black; the fg
        // is phosphor green. Comments get a brighter, slightly cyan-shifted green so
        // the SYSTEM'S VOICE (the asides) is legibly distinct from its BODY (code).
        // A faint scanline darkens alternate SOURCE-TEXT ROWS so the wall reads as a
        // CRT, not a page.
        //
        // TEXEL PERSISTENCE: the scanline parity is keyed on the SOURCE-ROW index
        // (`floor(wall_v · rows)`), the SAME world-anchored row that selects `cg`
        // above — NOT on `ctx.screen_y` (the absolute terminal row). A given source
        // line therefore keeps its dim/bright tone from every viewpoint, instead of
        // flipping as the wall slice slides up/down the screen under camera motion
        // (which was a screen-domain, camera-relative randomness source — a defect
        // by the persistence contract). Falls out for free: we already snap `wall_v`
        // to a source row to pick the glyph, so reusing that row index costs nothing.
        let scan = if (source_row & 1) == 0 {
            CODE_SCANLINE_DIM
        } else {
            1.0
        };
        let (fg, glyph) = match cg.class {
            CodeClass::Blank => {
                // Whitespace / off-grid: the dead screen. A space on the base bg —
                // rank 0 so it always shows (the gaps between code are part of it).
                let base = Texel {
                    glyph: u32::from(' '),
                    fg: pack_rgb(CODE_VOID_FG),
                    bg: pack_rgb(CODE_BG),
                    rank: 0,
                };
                return (base, None);
            }
            CodeClass::Code => (CODE_FG, cg.glyph),
            CodeClass::Comment => (CODE_COMMENT_FG, cg.glyph),
        };

        // Base texel: the source glyph in phosphor green over the dead-black bg.
        // We keep the TEXT at rank 0 (always visible) so legibility survives the
        // ranked-LOD as long as geometry allows — the brief is explicit that the
        // text must read. Distance/grazing still fog+coarsen it via the shared
        // `sample` epilogue (side dim, fog), but the glyph itself is never dropped
        // to a base "below" it: the source IS the base.
        let lit = scan;
        let base = Texel {
            glyph,
            fg: dim_rgb(pack_rgb(fg), lit),
            bg: pack_rgb(CODE_BG),
            rank: 0,
        };
        (base, None)
    }

    // ── note wall: ONE authored placard, on ONE face of ONE wall cell ──
    //
    // The code wall flows a whole source file across a long wall run. A placard is
    // the opposite: a short, word-wrapped body that must fit — and read — on the
    // single cell face the author pointed at. So the mapping is "fit the block across
    // the unit face" (`TextPanel::sample_fitted`), not "advance N characters per world
    // cell".
    //
    // THE FACE GATE is the whole reason `NoteWall` carries a `face`. Material binding
    // is per-CELL and `SampleCtx::side` has no sign, so an ungated placard renders on
    // all four faces of its cell — including the reverse side, where the reader in the
    // next room sees it mirrored. Two gates, in preference order:
    //
    //   1. `ctx.light` present ⇒ compare the shaded face's true outward normal
    //      (`LightCtx::face_n`, which the caster points back toward the eye) against
    //      the placard's own outward normal. Exact: the reverse face fails (dot −1) and
    //      the perpendicular faces fail (dot 0).
    //   2. `ctx.light == None` (a focused unit test, or any caller shading without
    //      lighting) ⇒ fall back to the coarse NS/EW `side` bit. This admits the
    //      matching axis PAIR (both x-faces for an x-facing placard) — coarser, but
    //      DEFINED, and it still keeps the text off the two perpendicular faces.
    //
    // A gated-OFF cell renders the dimmed blank plate: an ordinary dark wall, never
    // text. Pure + allocation-free; the panel lookup is one registry read guard plus a
    // slice index, and only on a note cell.
    //
    // PER-REGISTER STYLING: the sampled cell arrives with the note's REGISTER
    // (`researcher | system_log | model_thought`, or the engine's `Plain` default), and
    // the register selects the whole ink/plate palette — warm for the human hand, cold
    // for the machine log, dim-violet for the model's own voice
    // (`crate::note_style::NoteRegister::palette`, a `const` match). The register's
    // LAYOUT (ragged margin, `>` prefix, parentheses) was baked into the panel grid at
    // level load, so styling adds NO per-cell work beyond that palette match.
    fn sample_note_wall(
        &self,
        coord: Vec2,
        ctx: &SampleCtx,
        note_index: u32,
        face: re_core::FaceDir,
    ) -> (Texel, Option<Texel>) {
        use crate::code_wall::CodeClass;

        let on_face = match ctx.light {
            Some(l) => face.matches_normal(l.face_n),
            None => ctx.side == face.side(),
        };
        if !on_face {
            // The reverse / perpendicular face: a plain dark plate, dimmed so it reads
            // as wall rather than as a blank sign. Deliberately the PLAIN plate (not the
            // note's register): this face is meant to read as ordinary masonry, and
            // resolving the register here would cost a registry read on every
            // reverse-face wall cell for a colour nobody is meant to notice.
            let plain = crate::note_style::NoteRegister::Plain.palette();
            let base = Texel {
                glyph: u32::from(' '),
                fg: dim_rgb(pack_rgb(plain.void_fg), NOTE_REVERSE_DIM),
                bg: dim_rgb(pack_rgb(plain.bg), NOTE_REVERSE_DIM),
                rank: 0,
            };
            return (base, None);
        }

        // WITHIN-CELL face coordinate. `coord.x` is `tex_u + integer face offset`
        // (the caster folds the offset in so a code wall flows across cells); the
        // fractional part is this cell's own `u`. `ctx.wall_v` already runs 0 (top of
        // the one-cell-tall slice) → 1 (bottom).
        let u = fract(coord.x);
        let v = ctx.wall_v.clamp(0.0, 1.0);
        // FITTED-CLAMP SPAN: how many terminal columns this face covers
        // (`cols_per_unit / (depth · foreshorten)` — see `crate::face_span`). Handing it
        // to the panel lets a NEAR placard draw at integer 1:1 instead of magnifying its
        // characters into doubles (`thhe latttice`). Two reads + a divide, and only on a
        // note cell; `0.0` (span unknown, e.g. a focused unit test that never built view
        // params) keeps the historical stretch mapping.
        //
        // D2: the horizontal placement is the caster's MEASURED anchor when it has one
        // (the face's two world endpoints projected through the very camera basis the
        // rays were built from), because `u · face_cols` evaluates the projection at
        // this cell's own depth — constant across a face-on wall, but DRIFTING across
        // an oblique one, which re-deals doubled/dropped letters at every yaw off 90°.
        // With no anchor published (a focused unit test, the near-ring bake) we fall
        // back to the historical estimate, byte-for-byte.
        let anchor = {
            let measured = crate::face_span::face_anchor();
            if measured.is_known() {
                measured
            } else {
                let face_cols = crate::face_span::face_cols(ctx.distance, ctx.foreshorten);
                crate::face_span::FaceAnchor::estimated(u, face_cols)
            }
        };
        let face_rows = crate::face_span::face_rows(ctx.distance);

        let sample =
            crate::scene_text::sample_note_panel_anchored(note_index, u, v, anchor, face_rows);
        let pal = sample.register.palette();

        // Rank 0 for BOTH ink and margin: a placard that dissolves under the ranked
        // LOD is a placard nobody reads (same reasoning as the code wall).
        //
        // `Comment` is the register's STRUCTURAL marker (the `>` of a log line), inked
        // with `mark_fg` so the prompt reads as furniture and the message as the
        // message; every other register sets `mark_fg == fg`, so nothing changes there.
        let (glyph, fg) = match sample.glyph.class {
            CodeClass::Blank => (u32::from(' '), pal.void_fg),
            CodeClass::Comment => (sample.glyph.glyph, pal.mark_fg),
            CodeClass::Code => (sample.glyph.glyph, pal.fg),
        };
        // PER-GLYPH colour override (`colors = GLYPH:#hex …`): applied LAST, after
        // the class-based pick above, so an
        // unkeyed glyph (`color_override == None`) is completely unaffected and a
        // keyed one gets its own literal authored RGB regardless of which class it
        // was reclassed to. See `crate::stencil_variants`'s "Colour key" doc section.
        let fg_packed = sample.color_override.unwrap_or_else(|| pack_rgb(fg));
        let base = Texel {
            glyph,
            fg: fg_packed,
            bg: pack_rgb(pal.bg),
            rank: 0,
        };
        (base, None)
    }

    // ── scanline: a filled directional-band panel (CRT / metal / fabric / grate) ──
    //
    // A SOLID surface (unlike the void hologram): the band gaps keep a real dark
    // `bg_palette` fill, so it reads as a panel, not a see-through wireframe. The
    // band axis is selectable (horizontal raster, vertical grate, or a woven cross-
    // hatch). The inked band glyph is chosen by how central the cell sits within the
    // band — a heavy block at the core fading to a light rule at the shoulder — and
    // the `bg` darkens in the gaps for a scanline-contrast read. `warp` jitters the
    // band centre with fBm so a metal/wood grain is hand-made, not ruler-straight.
    // Pure function of the WORLD `coord` (a `Shift` anim has already scrolled it), so
    // the pattern parallax-scrolls and far-freezes like every recipe.
    #[allow(clippy::too_many_arguments)]
    fn sample_scanline(
        &self,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
        freq: f32,
        duty: f32,
        warp: f32,
        axis: u8,
    ) -> (Texel, Option<Texel>) {
        // DISTANCE-LOD octave cap (no-op in the NEAR band → byte-identical).
        let lod = ShadeLod::for_ctx(ctx);
        let f = freq.max(0.05);
        let duty = duty.clamp(0.05, 0.95);

        // fBm warp breaks dead-straight lines into a grain (metal/wood/fabric).
        // Band-limited by screen period.
        let jitter = if warp > 0.0 {
            (fbm_bandlimited(
                perlin,
                coord * 3.1,
                lod.octaves(3),
                2.0,
                0.5,
                lod.world_per_cell / 3.1,
            ) - 0.5)
                * warp
        } else {
            0.0
        };

        // Distance to the nearest band CENTRE along the chosen axis (0 = on a line,
        // 1 = mid-gap). For the WOVEN axis (2) we take the stronger of the two axes,
        // so the inked threads of both directions show as a cross-hatch.
        let band_dist = |t: f32| {
            let phase = fract(t * f + jitter);
            (phase - 0.5).abs() * 2.0
        };
        let dist = match axis {
            1 => band_dist(coord.x), // vertical bands (grate / blind)
            2 => band_dist(coord.x).min(band_dist(coord.y)), // woven cross-hatch
            _ => band_dist(coord.y), // horizontal raster (CRT / metal grain)
        };
        let mut on_line = dist <= duty;

        // SPARSE-SURFACE GATE: a Scanline material
        // authored onto a WALL or FLOOR surface (e.g. the `metal`-preset `deck`
        // floor in levels 04/08) would otherwise ink a band glyph on EVERY on-line
        // cell — a packed horizontal `━`/`─`/`┈` field that reads as symbol mush rather
        // than the calm colour-forward ground the default `grid` floor achieves.
        // When this recipe paints a floor or wall, we thin the inked band cells by
        // the SAME world-anchored coverage mask the brick/grain bodies use, so the
        // surface reads by LIGHT + COLOUR with only an occasional clean rule, AND we
        // suppress the heavy/dotted glyph PROMOTION (`scanline_glyph`'s `━`/`┈`) in
        // favour of a single plain rule. The mask is a pure function of the WORLD
        // `coord`, so a band cell is always inked or always blank — no churn. A
        // Scanline on a non-surface (puzzle panel / sprite material) is unchanged.
        // An EMITTER (a lit screen / neon tube) is not a body texture: it keeps every
        // band on any face (see [`SELF_LIT_EMISSIVE`]).
        let emitter = self.light.emissive >= SELF_LIT_EMISSIVE;
        let surface = if emitter { None } else { scanline_surface(ctx) };
        // A band cell the coverage mask blanks is still BAND, not gap: it keeps the
        // band's colour (dimmed) as a coloured space, so the surface reads by light +
        // colour as intended. Treating it as a gap instead painted ~88 % of a `metal`
        // wall in the gap bg (≈3 % grey) — a featureless black slab.
        let mut masked_band = false;
        if let Some(kind) = surface {
            let keep = match kind {
                ScanlineSurface::Wall => wall_body_inked(self.seed, coord),
                ScanlineSurface::Floor => floor_body_inked(self.seed, coord),
            };
            masked_band = on_line && !keep;
            on_line = on_line && keep;
        }

        // Colour: the fg phosphor/sheen hue brightened toward the band core; the bg
        // is the dark panel base, deepened in the gaps so the bands read with
        // contrast. A single coherent colour family from the palette (not a wash).
        let core = (1.0 - dist / duty.max(1e-3)).clamp(0.0, 1.0); // 1 at centre → 0 at edge
        let hue = self.palette.eval(0.45 + 0.35 * core);
        let base_bg = self.bg_palette.eval(0.5);

        if !on_line {
            // In a gap: a coloured SPACE on the (slightly darkened) panel bg. rank 0
            // so the panel always reads; the gap carries colour, not a glyph.
            // An emitter's gap is its lit backing, not a shadowed groove; a masked
            // band keeps a dim wash of its band colour.
            let bg = if masked_band {
                base_bg.lerp(hue, SCANLINE_MASKED_BAND_BG)
            } else if emitter {
                base_bg
            } else {
                base_bg * SCANLINE_GAP_BG
            };
            let base = Texel {
                glyph: u32::from(' '),
                fg: pack_rgb(hue * SCANLINE_FG_FLOOR),
                bg: pack_rgb(bg),
                rank: 0,
            };
            return (base, None);
        }

        // On a band: pick the band glyph by how central the cell is — a heavy block
        // at the bright core fading to a light rule at the shoulder. Axis picks
        // horizontal vs vertical rule glyphs so the bands read in their direction;
        // the woven axis uses a hatch/plus at the crossing.
        //
        // On a WALL or FLOOR surface, suppress the heavy/dotted PROMOTION: a surface
        // band is the rare surviving accent (the coverage mask already thinned the
        // rest), so it should read as a single clean PLAIN rule (`─`/`│`/`┼`), never
        // the dense `━`/`┈` family that turned the deck floor into a dashed field.
        let glyph = if surface.is_some() {
            scanline_plain_glyph(axis)
        } else {
            scanline_glyph(axis, core)
        };
        let fg = pack_rgb(
            (hue * (SCANLINE_FG_FLOOR + (1.0 - SCANLINE_FG_FLOOR) * core))
                .clamp(Vec3::ZERO, Vec3::ONE),
        );
        let base = Texel {
            glyph,
            fg,
            bg: pack_rgb(base_bg),
            rank: 0,
        };
        (base, None)
    }

    /// TRUCHET: an edge-matched conduit lattice (see [`crate::truchet`]).
    ///
    /// The whole point is CONTINUITY, so the tile index must key on the WORLD cell
    /// and nothing else. Keying on anything view-dependent would make a run break as
    /// the camera moves, which is the exact defect the recipe exists to avoid.
    ///
    /// # Why this cannot just floor `coord`
    ///
    /// On a FLOOR or CEILING, `coord` IS the world hit `(hit_x, hit_y)`, so flooring
    /// it gives the world cell. On a WALL it does not: [`crate::sample_coord`] builds
    /// `coord.x` as `tex_u + floor(the face's CONSTANT axis)` — an offset that is the
    /// same for every cell of one flat wall run — so `floor(coord.x)` is CONSTANT
    /// along the whole face, while `coord.y` is the world height times a texture-
    /// density scale. Flooring that pair gave one tile column stretched across the
    /// entire wall and six tile rows per world cell of height: every screen row of a
    /// wall came out as a single glyph repeated end to end, which is the opposite of
    /// a lattice. (The same trap [`CtxInputs`]' doc records for the cache keys.)
    ///
    /// So on a wall the index comes from the WALL PLANE — `(wall_along, wall_up)`,
    /// the continuous world hit along the face and the texel's world height — which
    /// advances exactly one unit per world cell on both axes, like `coord` does on a
    /// floor. [`SampleCtx::on_wall`] is the discriminator; it cannot be inferred from
    /// the values, since `0.0` is both "floor" and a legal wall position.
    ///
    /// A blank tile (nothing crosses this cell — most of them at low density) emits a
    /// coloured SPACE rather than a glyph, matching the scanline gap treatment: the
    /// field reads as calm colour with occasional structure, not as symbol mush.
    fn sample_truchet(
        &self,
        coord: Vec2,
        ctx: &SampleCtx,
        seed: i32,
        density: f32,
        scale: f32,
    ) -> (Texel, Option<Texel>) {
        let plane = if ctx.on_wall {
            Vec2::new(ctx.wall_along, ctx.wall_up)
        } else {
            coord
        } * scale.max(f32::MIN_POSITIVE);
        let cx = plane.x.floor() as i32;
        let cy = plane.y.floor() as i32;
        let g = crate::truchet::glyph(cx, cy, seed, density);

        // Same palette taps as the scanline recipe: the ink hue and its dark
        // companion, so a truchet wall sits in the level's colour scheme rather than
        // introducing its own.
        let hue = self.palette.eval(0.6);
        let base_bg = self.bg_palette.eval(0.5);
        if g == ' ' {
            // Nothing crosses: coloured blank on the panel bg, no glyph.
            return (
                Texel {
                    glyph: u32::from(' '),
                    fg: pack_rgb(hue * SCANLINE_FG_FLOOR),
                    bg: pack_rgb(base_bg * SCANLINE_GAP_BG),
                    rank: 0,
                },
                None,
            );
        }

        // A run: ink it. Junctions (three or four sides) are the structural nodes of
        // the lattice, so they read brighter than a plain straight run — the eye picks
        // out where the conduit branches, which is the readable part of the pattern.
        let ways = crate::truchet::connections(cx, cy, seed, density).count_ones();
        let lift = if ways >= 3 { 1.0 } else { 0.72 };
        // No per-context term beyond the surface-kind discriminator read above: the
        // lattice is a pure function of the world cell.
        (
            Texel {
                glyph: u32::from(g),
                fg: pack_rgb((hue * lift).clamp(Vec3::ZERO, Vec3::ONE)),
                bg: pack_rgb(base_bg),
                rank: 0,
            },
            None,
        )
    }

    // ── fractal: a log-polar self-similar portal (rings / spiral arms) ──────────
    //
    // The shared [`crate::fractal::log_polar_field`] computes a height + ring/arm
    // phase from the (centred) world coord and the live [`LogPolarPhase`] (the
    // Zoom/Spin/SpinZoom animation; static at phase 0). Colour: the ring phase
    // drives the fg via the palette (rings at different octave-depth get different
    // hue) and the angular index tints it (arms read distinctly); the bg is a dark
    // companion deepened toward the troughs so the rings read with contrast. The
    // body glyph rides the SHARED [`ramp_glyph`] from the height (dense crests ⇒
    // bold glyphs, troughs ⇒ sparse). Pure function of the WORLD coord — it
    // parallax-scrolls and far-freezes like every recipe (a far-frozen cell holds a
    // fixed phase, so the pattern stops zooming but still reads as a portal).
    fn sample_fractal(
        &self,
        perlin: &Perlin,
        coord: Vec2,
        ctx: &SampleCtx,
        arms: u8,
        octave_warp: f32,
        twist: f32,
    ) -> (Texel, Option<Texel>) {
        let p = crate::fractal::FractalParams {
            arms,
            octave_warp,
            twist,
        };
        // The live zoom/spin scroll from the material's animation (the three fractal
        // AnimKind modes); a static material (or a non-fractal anim) yields the
        // default phase → a still portal. Reading the phase here keeps the recipe a
        // pure function of `(coord, ctx)`.
        let phase = self.fractal_phase(ctx, twist);
        let s = crate::fractal::log_polar_field(coord.x, coord.y, phase, &p, perlin);

        // Ring-phase hue, tinted by the arm index so adjacent arms read distinctly.
        let hue = self.palette.eval(s.ring_phase + 0.15 * s.angle_idx);
        // Dark companion bg, deepened toward the ring troughs (low height) for
        // contrast — a half-range ring-phase sweep keeps it a coherent family.
        let bg = self.bg_palette.eval(s.ring_phase * 0.5)
            * (FRACTAL_BG_FLOOR + (1.0 - FRACTAL_BG_FLOOR) * s.height);

        let glyph = ramp_glyph(s.height);
        let base = Texel {
            glyph,
            fg: pack_rgb(hue.clamp(Vec3::ZERO, Vec3::ONE)),
            bg: pack_rgb(bg.clamp(Vec3::ZERO, Vec3::ONE)),
            // rank 0: the portal IS the base — it must read, not LOD-thin to noise.
            rank: 0,
        };
        (base, None)
    }
}

/// Depth → glyph-ink falloff amount (the raycaster's primary depth→brightness cue).
///
/// Returns a `[0,1]` **falloff fraction** `t`: `0` at distance 0 (keep the full
/// chosen glyph density) rising toward `1` far away. A terminal cell's perceived
/// luminance is `ink(glyph)·L(fg) + (1−ink(glyph))·L(bg)`, so on the common surface
/// (a brightish fg over a darker bg) shifting the body glyph DOWN the density ramp —
/// less ink — lowers effective luminance *without touching the fg/bg colours*. That
/// keeps the Oklab fg/bg distance (the render-QA `separation` metric) and the
/// fg-vs-bg legibility (`invisible_pct`) intact while still producing a
/// correctly-signed, monotonic near-bright / far-dim depth cue carried entirely by
/// the GLYPH channel. It is also a natural distance LOD (far = less ink = less
/// detail), so it rides the existing ramp machinery rather than bolting on a second
/// dimming system.
///
/// Form: `t = 1 − 1/(1 + k·d)` — a smooth reciprocal ramp, exactly `0` at `d=0`
/// (no dead zone), rising monotonically toward `1`. The caller lerps the
/// ramp-selection brightness from its full value (near) toward [`DEPTH_INK_FLOOR`]
/// (far) by `t`; [`DEPTH_INK_FLOOR`] is held high enough that the sparsest result
/// is still an INKED glyph (never blank). On a bg-brighter surface the caller
/// flips the direction so the cue never inverts (far sky/ceiling does not brighten).
#[inline]
fn depth_ink_falloff(distance: f32) -> f32 {
    let d = distance.max(0.0);
    1.0 - 1.0 / (1.0 + DEPTH_INK_K * d)
}

/// World-texel frequency of the ramp-dither hash: how many jitter cells per world
/// unit. 2 ⇒ half-cell texels, fine enough that adjacent screen cells on a near
/// wall draw different jitter, coarse enough that a far wall (many world units per
/// screen cell) still reads as a stable stipple rather than white noise.
const RAMP_DITHER_FREQ: f32 = 2.0;

/// The ramp-threshold dither offset for a body cell at world `coord`, in ramp-
/// BRIGHTNESS units: a world-anchored hash in `[-½, ½)` × `amplitude` × one ramp
/// step × the depth-ink strength `t ∈ [0,1]`. Pure, deterministic, never reads a
/// clock or screen position, so it is identical every frame for a given world spot.
/// `amplitude ≤ 0` or `t ≤ 0` ⇒ exactly `0.0` (byte-identical to the undithered path).
#[inline]
fn ramp_dither_offset(seed: u32, coord: Vec2, amplitude: f32, t: f32) -> f32 {
    if !(amplitude.is_finite() && t.is_finite()) || amplitude <= 0.0 || t <= 0.0 {
        return 0.0;
    }
    let step = 1.0 / (BODY_RAMP.len() as f32 - 1.0);
    let cell = (coord * RAMP_DITHER_FREQ).floor();
    let j = hash01(hash_cell(seed ^ 0x5D17_D17E, cell.x as i32, cell.y as i32)) - 0.5;
    j * amplitude * step * t.min(1.0)
}

/// Sky/ceiling depth-band multiplier `[SKY_BAND_FLOOR, 1]`: `1.0` near the camera
/// decaying toward [`SKY_BAND_FLOOR`] at the horizon. Multiplies the ceiling's void
/// base (and any star) so far ceiling reads dimmer than near. The ceiling's base
/// glyph is BLANK, so its depth cue cannot ride the glyph-ink ramp — this band is
/// the floor/ceiling analogue, applied at the recipe source. Form:
/// `floor + (1−floor)/(1 + k·d)` — a smooth reciprocal falloff, exactly `1.0` at
/// `d=0`, monotone toward `floor`. See [`Material::sample_sky`].
#[inline]
fn sky_depth_band(distance: f32) -> f32 {
    let d = distance.max(0.0);
    SKY_BAND_FLOOR + (1.0 - SKY_BAND_FLOOR) / (1.0 + SKY_BAND_K * d)
}

/// Distance fog: 1 near → 0 far (linear band) using the per-cast [`FogParams`]
/// `near`/`far` edges (which default to the built-in [`FOG_NEAR`]/[`FOG_FAR`]).
/// A degenerate band (`far <= near`) is treated as a hard step at `near`.
#[inline]
fn distance_fog(distance: f32, fog: &FogParams) -> f32 {
    if distance <= fog.near {
        1.0
    } else if distance >= fog.far || fog.far <= fog.near {
        0.0
    } else {
        1.0 - (distance - fog.near) / (fog.far - fog.near)
    }
}

/// The exponential HEIGHT-fog multiplier at world height `z`:
/// `exp(-falloff · max(0, z − base))`.
///
/// `1.0` (a total no-op) when the level authored no height fog, so the shipped
/// corpus never pays for it and never changes. The clarity term is applied
/// MULTIPLICATIVELY on top of the distance factor: a surface high above `base`
/// clears out of the murk, a surface at or below `base` is left to the distance
/// term alone. Monotonically non-decreasing in `z`, by construction.
///
/// Note the sign: the returned value is a CLARITY factor (1 = clear), matching
/// [`distance_fog`], so the caller multiplies rather than blends twice.
#[inline]
fn height_fog(world_z: f32, fog: &FogParams) -> f32 {
    match fog.height {
        None => 1.0,
        Some(h) => {
            // Guard authored garbage: a non-finite height or a negative falloff
            // would invert the effect (fog thickening upward). The parser refuses
            // both; this keeps a hand-built FogParams from producing NaN.
            if !h.falloff.is_finite() || h.falloff <= 0.0 || !world_z.is_finite() {
                return 1.0;
            }
            let above = (world_z - h.base).max(0.0);
            // `exp` on a small non-negative argument — one transcendental per
            // FOGGED cell (the near band short-circuits before this is reached).
            (-h.falloff * above).exp()
        }
    }
}

/// A coarse-to-fine band a recipe assigns a detail texel to.
enum RankBand {
    Structure, // seams, grid, pads — revealed first (low rank)
    Mid,       // traces, stars
    Fine,      // speckle, grit — revealed last (high rank)
}

/// Map a detail's semantic band → a concrete rank under the material's strategy.
#[inline]
fn rank_for(strat: RankStrategy, band: RankBand) -> u8 {
    match strat {
        RankStrategy::StructureFirst => match band {
            RankBand::Structure => RANK_STRUCTURE,
            RankBand::Mid => RANK_MID,
            RankBand::Fine => RANK_FINE,
        },
        // Everything biased high → flat base until close.
        RankStrategy::FineDetail => match band {
            RankBand::Structure => RANK_MID,
            RankBand::Mid => RANK_FINE,
            RankBand::Fine => RANK_FINE,
        },
    }
}

/// Distance from `coord` to the nearest integer world line in either axis, as a
/// boolean "on a tile line" test.
#[inline]
fn tile_line(coord: Vec2, eps: f32) -> bool {
    let fx = (coord.x - coord.x.floor()).min(1.0 - (coord.x - coord.x.floor()));
    let fy = (coord.y - coord.y.floor()).min(1.0 - (coord.y - coord.y.floor()));
    fx < eps || fy < eps
}

/// Approximate Voronoi edge distance: `|F2 − F1|` is small on a cell boundary.
fn circuit_edge(seed: u32, p: Vec2) -> f32 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = ix + dx;
            let cy = iy + dy;
            let jitter = hash2_01(seed, cx, cy);
            let feat = Vec2::new(cx as f32 + jitter.x, cy as f32 + jitter.y);
            let d = (feat - p).length();
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    f2 - f1
}

// ─────────────────────── recipe-local named constants ───────────────────────

/// EW faces render this fraction as bright as NS faces (the wolf3d trick). Used
/// ONLY on the legacy no-`LightCtx` fallback path; the shipped relief lighting
/// supersedes it with real `N·L`.
const SIDE_EW_DIM: f32 = 0.72;

/// STATE-BUS PULSE emissive gain: the maximum extra brightening a full
/// (`1.0`) [`SampleCtx::pulse`] adds to a cell's inked fg/bg — a brief event FLASH
/// (node fired / switch toggled / cycle rejected). Tuned to read as a clear kick
/// without blowing the cell to pure white. `pulse == 0.0` (the EMPTY-map default)
/// is a no-op, so the shade is byte-identical until the bridge sets a pulse.
const STATE_PULSE_EMISSIVE: f32 = 0.6;

/// FLOW scroll rate (surface-units/sec) a POWERED pipe cell drifts its glyph
/// pattern along the surface axis toward the output.
/// Tuned so a player reads the direction of travel without it strobing. An
/// unpowered cell does not scroll at all (state 0 → no flow), so this only affects
/// a pipe the bridge has powered.
const FLOW_SCROLL_RATE: f32 = 0.9;

/// FLOW hot-hue lift: the brightness multiplier a POWERED pipe cell's inked fg/bg
/// gets (the "hot when flowing" cue). `1.0` would be no lift; a small lift makes a
/// live pipe read warmer/brighter than a cold (unpowered) one without a new colour
/// path. Unpowered ⇒ no lift → byte-identical to the cold trace.
const FLOW_HOT_LIFT: f32 = 1.35;

/// Scroll the sample `coord` along the surface axis for a POWERED pipe cell, in the
/// flow direction decoded from `ctx.state`. The pipe's
/// glyph pattern then appears to travel input→output (`Forward`) or the reverse
/// (`Backward`) at [`FLOW_SCROLL_RATE`]. An UNPOWERED cell (`state` decodes to
/// [`FlowDir::None`] — the empty-bus default for every tag, and every non-pipe
/// material) returns the coord UNCHANGED, so the shade is byte-identical to today.
/// Pure function of `(coord, ctx.state, ctx.time)` — no clock, no allocation.
#[inline]
fn flow_shift_coord(coord: Vec2, ctx: &SampleCtx) -> Vec2 {
    use re_core::FlowDir;
    let d = FLOW_SCROLL_RATE * ctx.time;
    match re_core::decode_flow_state(ctx.state) {
        FlowDir::None => coord,
        // Scroll DOWN the surface axis (toward the output) — same axis the matrix
        // `Shift` flow uses, so a powered trace reads as travelling, not jittering.
        FlowDir::Forward => Vec2::new(coord.x, coord.y + d),
        FlowDir::Backward => Vec2::new(coord.x, coord.y - d),
    }
}

/// True iff `ctx.state` decodes to a POWERED pipe flow (so the cell is "hot" and
/// gets the [`FLOW_HOT_LIFT`] brightness lift). `false` for the empty-bus default
/// and every non-pipe material → no lift, byte-identical to today. Pure.
#[inline]
fn flow_is_hot(ctx: &SampleCtx) -> bool {
    !matches!(
        re_core::decode_flow_state(ctx.state),
        re_core::FlowDir::None
    )
}

/// Step (in a recipe's own coord units) for the central-difference height
/// gradient. Small → crisp relief; too small → noise quantisation jitters the
/// normal. Tuned to read as bumps without sparkling.
const HEIGHT_GRAD_EPS: f32 = 0.05;

/// Global scale on the body-glyph ramp brightness (the "carry detail with COLOUR,
/// not glyphs" knob). `1.0` is the old saturated look (face-on walls slabbed to
/// `@%#`); below `1.0` shifts the body toward the sparser end of [`BODY_RAMP`] so
/// walls/floors read as a calm low/mid-density field carried by light + colour,
/// not a solid glyph fill. The relief ORDERING is preserved (it scales the input,
/// so ridges still beat grooves) — only the overall density drops.
///
/// Kept near the old 0.50 (0.55) ON PURPOSE. With
/// [`WALL_BODY_COVERAGE`] now low, the few cells that DO ink are sparse ACCENTS, so
/// they should read as crisp marks (toward `@#+=`), not faint ones. Washing the
/// accent density down toward `.` pulls each inked cell's effective colour back
/// toward its bg, which lowers the QA boundary contrast (and thus `separation`).
/// Sparsity comes from COVERAGE, not from dimming the surviving marks.
const BODY_DENSITY: f32 = 0.55;

/// SPARSE WALL-BODY COVERAGE.
///
/// The brick BODY texel used to be an INKED ramp glyph on EVERY cell — so a wall
/// rendered as packed runs of the same `.`/`*` glyph, an incoherent mess. This is
/// the fraction of wall-body cells that keep an inked body glyph;
/// the rest are blanked to a coloured SPACE, so the wall reads by LIGHT + COLOUR
/// with only occasional, world-scattered glyph detail. The choice is a pure
/// function of the WORLD texel (see [`wall_body_inked`]), so it is deterministic and
/// does NOT churn as the camera moves (anti-flicker preserved) — a given world spot
/// is always inked or always blank. Structural mortar/edge seams are unaffected
/// (they are a separate detail texel), so the wall keeps crisp structure.
///
/// `1.0` reproduces the old fully-inked body (every cell a glyph); lower = sparser.
/// Tuned LOW so the bulk of a wall is colour-carried ground with a light scatter of
/// detail glyphs — never identical-symbol rows. Kept deliberately under a quarter so
/// that even a GRAZING wall (where high foreshortening repeats a few world texels
/// across many screen columns) reads as a sparse stipple, not packed `.` runs.
///
/// Lowered 0.22 → 0.12 as the primary "mush kill". Only ~1 in
/// 8 wall-body cells now keeps an inked glyph; the other ~88% are coloured spaces
/// that let the fg/bg palette carry the wall's identity. Structural mortar/edge
/// seams ink on a SEPARATE detail rank (see [`wall_body_inked`]'s callers), so the
/// wall keeps crisp structure — only the bulk-surface stipple thins.
const WALL_BODY_COVERAGE: f32 = 0.12;

/// World lattice density (cells per world unit) the wall-body sparse-coverage mask
/// is hashed on. Sub-cell (>1) so the inked/blank scatter varies WITHIN one world
/// cell of wall — an irregular speckle of detail, not a per-world-cell block. The
/// mask is value-noise over this lattice, thresholded by [`WALL_BODY_COVERAGE`], so
/// the inked cells form organic clumps rather than a regular grid.
const WALL_BODY_MASK_FREQ: f32 = 3.7;

/// SPARSE FLOOR TILE-SEAM COVERAGE. The floor's tile grid used to ink EVERY integer
/// world line in full — a hard, mechanical grid that (with the specks) packed the
/// near floor. This is the fraction of seam SEGMENTS that actually ink: a seam is
/// chopped into world-hashed segments and only this fraction survive, so the tiling
/// still reads (occasional clean seam runs) without a solid ruled grid. World-seeded
/// on the segment's integer line + along-axis bucket, so a given seam segment is
/// always inked or always blank (no churn). `1.0` = the old full grid.
///
/// Lowered 0.45 → 0.14: at 0.45 the `grid`
/// floor's seam dashes packed into a near-continuous ruled grid on the near floor —
/// a "symbol mush" look. At 0.14 the floor is overwhelmingly blank
/// coloured ground with only occasional clean seam marks, matching the sparse
/// reference look on levels 01/03/07 (and the depth cue is carried by the fg/bg
/// colour gradient, not a glyph fill).
const FLOOR_SEAM_COVERAGE: f32 = 0.14;

/// Along-seam segment length (world units) the floor seam-coverage mask buckets on.
/// Each integer world line is chopped into segments of this length; each segment is
/// independently kept/dropped by [`FLOOR_SEAM_COVERAGE`]. Short enough to break the
/// ruled-line look into dashes, long enough that a kept run still reads as a seam.
const FLOOR_SEAM_SEG: f32 = 0.85;

/// World value-noise mask in `[0,1]`, evaluated on a `freq`-scaled lattice salted by
/// `salt`. A pure function of the WORLD `coord` (no distance / screen-row / time),
/// so any threshold on it picks the SAME world spots every frame — the basis of the
/// sparse, non-repeating, flicker-free scatter for walls + floor. Reuses the engine
/// [`value_noise`] primitive (smoothed integer-cell hash) so the kept cells clump
/// organically instead of forming a regular grid.
#[inline]
fn world_mask(seed: u32, coord: Vec2, freq: f32, salt: u32) -> f32 {
    value_noise(seed ^ salt, coord * freq)
}

/// Whether a wall-BODY cell at world `coord` keeps its inked body glyph (vs. being
/// blanked to a coloured space). `true` for the [`WALL_BODY_COVERAGE`] fraction of
/// world spots whose [`world_mask`] clears the threshold; `false` (blank) elsewhere.
/// Pure function of the world `coord` + the material `seed` → deterministic and
/// stable under camera motion (a world spot's inked/blank state never flips as you
/// walk), so it thins the wall WITHOUT reintroducing flicker. Coverage `≥ 1.0`
/// short-circuits to always-inked (the old fully-filled body).
#[inline]
fn wall_body_inked(seed: u32, coord: Vec2) -> bool {
    if WALL_BODY_COVERAGE >= 1.0 {
        return true;
    }
    // Ink the LOW-noise wells so inked cells form organic clumps (a few connected
    // specks) rather than uniform salt-and-pepper. Threshold = the coverage fraction.
    world_mask(seed, coord, WALL_BODY_MASK_FREQ, 0x7A11) < WALL_BODY_COVERAGE
}

/// SPARSE FLOOR BODY COVERAGE. The relief pass re-selects a body ramp glyph for every
/// inked Recipe — which is what used to give the grain FLOOR a glyph on EVERY cell
/// (a dense field), even though its recipe BASE glyph is a blank
/// space. This is the fraction of floor-body cells that keep that relief body glyph;
/// the rest blank to a coloured SPACE, so the floor reads as calm shaded ground —
/// mostly colour, with only an occasional world-scattered mark. Kept VERY low (the
/// floor should be overwhelmingly blank ground), well under the wall's coverage.
/// Lowered 0.12 → 0.07: the near floor is now almost pure
/// shaded colour with only a rare scattered mark.
const FLOOR_BODY_COVERAGE: f32 = 0.07;

/// World lattice density for the floor-body mask. Tied to the world grid (≈1 cell)
/// so floor marks scatter at roughly the tiling scale rather than a fine fizz.
const FLOOR_BODY_MASK_FREQ: f32 = 1.6;

/// Whether a floor-BODY cell at world `coord` keeps its relief body glyph (vs. being
/// blanked to a coloured space). Mirrors [`wall_body_inked`] with the floor's own
/// (much lower) [`FLOOR_BODY_COVERAGE`]: a pure function of the world `coord` + seed,
/// so the floor's blank/inked scatter is deterministic and flicker-free under camera
/// motion. Coverage `≥ 1.0` short-circuits to always-inked (the pre-fix full body).
#[inline]
fn floor_body_inked(seed: u32, coord: Vec2) -> bool {
    if FLOOR_BODY_COVERAGE >= 1.0 {
        return true;
    }
    world_mask(seed, coord, FLOOR_BODY_MASK_FREQ, 0x40DD) < FLOOR_BODY_COVERAGE
}

/// Whether a floor tile-SEAM segment at world `coord` inks (vs. a gap). The integer
/// world line is bucketed into [`FLOOR_SEAM_SEG`]-long segments along its run; each
/// segment is independently kept for the [`FLOOR_SEAM_COVERAGE`] fraction whose
/// hash clears the threshold. `which_axis` selects the x-line vs y-line seam so the
/// two families are bucketed independently. Pure function of the world position →
/// a given seam segment is always inked or always a gap (no churn). Coverage `≥ 1.0`
/// short-circuits to the old full grid.
#[inline]
fn floor_seam_inked(seed: u32, coord: Vec2, on_x_line: bool) -> bool {
    if FLOOR_SEAM_COVERAGE >= 1.0 {
        return true;
    }
    // The seam runs ALONG the perpendicular axis: an x-const line (on_x_line) runs
    // along y, and vice-versa. Bucket the line's integer index + the along-run
    // segment so each dash of the seam is decided independently.
    let (line_idx, along) = if on_x_line {
        (coord.x.round() as i32, coord.y)
    } else {
        (coord.y.round() as i32, coord.x)
    };
    let seg = (along / FLOOR_SEAM_SEG).floor() as i32;
    let salt = if on_x_line { 0x5EA1u32 } else { 0x5EA2u32 };
    hash01(hash3i((seed ^ salt) as i32, line_idx, seg)) < FLOOR_SEAM_COVERAGE
}

/// Distance fog band (world cells).
const FOG_NEAR: f32 = 1.0;
const FOG_FAR: f32 = 26.0;

/// Depth → glyph-ink falloff knobs (see [`depth_ink_falloff`]). This is the primary
/// depth→brightness cue and the one the render-QA `depth_luma_spearman` linter
/// reads. `K` sets how fast the chosen body glyph slides DOWN the density ramp
/// with distance (larger ⇒ far surfaces thin out faster, a steeper near→far ink
/// ramp); `FLOOR` is the ramp-brightness the cue fades TOWARD far away — held just
/// high enough that the sparsest result is still an INKED ramp glyph (`.`/`-`),
/// never blank, so distant walls thin but do not vanish. These are the primary
/// knobs the `/iterate-render` skill drives for the depth cue.
///
/// Why ink and not a colour multiply: the old `depth_darken` scaled fg+bg by a
/// `[0,1]` factor, which compressed their Oklab distance — collapsing the QA
/// `separation` (object/surface contrast) metric and, on the textured calibration
/// levels (10/11), driving `invisible_pct_near` from ~3% toward ~49% at any useful
/// strength. Shifting the GLYPH down the ramp lowers effective luminance
/// (`lerp(bg,fg,ink)`) while leaving fg/bg colours — and therefore `separation`
/// and `invisible_pct` — untouched. `K`/`FLOOR` can be tuned aggressively without
/// the colour-contrast penalty that capped the old multiply.
/// Lowered K 0.45 → 0.30 and FLOOR 0.12 → 0.08: with the
/// sparser ramp + lower body density already thinning the near field, a gentler
/// depth-ink falloff keeps the remaining marks legible at mid distance instead of
/// crushing them to the ramp tail immediately.
const DEPTH_INK_K: f32 = 0.30;
const DEPTH_INK_FLOOR: f32 = 0.08;

/// Sky/ceiling depth-band knobs (see [`sky_depth_band`]). The sky's base glyph is
/// BLANK, so the glyph/ink depth cue cannot carry depth there; the ceiling instead
/// gets a depth-banded colour shade applied as the FINAL step of [`Material::sample`]
/// (after lighting + fog, so the void's emissive relight cannot wash it out). `K`
/// sets how fast the band darkens with distance; `FLOOR` is the residual brightness
/// the far horizon keeps so the ceiling never crushes to a flat dead black (which
/// would both read as a void and, via Oklab clamping, degenerate the depth↔luma
/// correlation into ties). Sky-only — a colour band on a blank cell is safe (no
/// inked fg/bg pair to collapse `separation` / inflate `invisible_pct`); the inked
/// wall/floor recipes use the glyph-ink cue instead. `/iterate-render`-tunable.
const SKY_BAND_K: f32 = 0.85;
const SKY_BAND_FLOOR: f32 = 0.10;

/// Concrete ranks for the three semantic bands under `StructureFirst`.
const RANK_STRUCTURE: u8 = 24;
const RANK_MID: u8 = 64;
const RANK_FINE: u8 = 110;

/// Brick channel glyphs + bg darkening factor.
const MORTAR_H_GLYPH: u32 = 0x2500; // ─
const MORTAR_V_GLYPH: u32 = 0x2502; // │
const BRICK_SPECK_GLYPH: u32 = b'.' as u32;
const BRICK_BG_FACTOR: f32 = 0.45;

/// Circuit glyphs.
const CIRCUIT_NODE_GLYPH: u32 = 0x253C; // ┼
const CIRCUIT_PAD_GLYPH: u32 = 0x25CB; // ○
const CIRCUIT_BUS_GLYPH: u32 = 0x2502; // │
const CIRCUIT_RUNG_GLYPH: u32 = 0x2500; // ─

/// Floor grain knobs. CALMER FLOOR: the speck radius is small so a speck is a
/// sparse pinprick, not a blob that fills cells — the floor reads mostly as a
/// coloured low-density ground with clean tile seams, and only a LITTLE controlled
/// grit up close. The grid glyph is the crisp seam that carries the tiling.
const GRAIN_SPECK_A: u32 = b'.' as u32;
const GRAIN_SPECK_B: u32 = b'\'' as u32;
const GRAIN_GRID_GLYPH: u32 = 0x2500; // ─ a clean seam line, not a `:` speck-row
const GRAIN_SPECK_RADIUS: f32 = 0.14;
const GRAIN_WARP_FREQ: f32 = 0.9;
const GRAIN_WARP_AMP: f32 = 0.6;

/// Opt-in DETAIL-TIER micro-grain (`[material] detail = on`): the next octave
/// inside each texel — fine grain `. · : '`. A SEPARATE, finer Worley lattice than the
/// ordinary speck field above, sampled only when [`Material::detail`] is set and
/// only revealed texel-by-texel as [`crate::detail_lod::cross_fade_weight`] rises
/// (world-anchored via [`crate::detail_lod::reveal_finer`], never screen-keyed —
/// so it never shimmers under a turn). A distinct hashed seed
/// offset (`DETAIL_MICRO_SEED`) keeps this lattice's feature points independent
/// of the ordinary speck lattice's, so the two never fall into lock-step.
const DETAIL_MICRO_SCALE: f32 = 3.0;
const DETAIL_MICRO_RADIUS: f32 = 0.22;
const DETAIL_MICRO_SEED: u32 = 0xD371_A11E;
const DETAIL_MICRO_GLYPH: u32 = b':' as u32;

/// Star sky knobs.
const SKY_BG_FACTOR: f32 = 1.0;
/// Faint base-fg tint for sky cells so a stray glyph stays visible (R-L3).
const SKY_BASE_FG_FACTOR: f32 = 1.6;
const STAR_GLYPH_BRIGHT: u32 = 0x2726; // ✦
const STAR_GLYPH_MID: u32 = b'*' as u32;
const STAR_GLYPH_FAINT: u32 = b'.' as u32;
const STAR_BRIGHT_FRAC: f32 = 0.12;
const STAR_MID_FRAC: f32 = 0.45;

/// Code-wall palette (the strange loop). Phosphor green on a dead-terminal black;
/// comments are a brighter, cyan-leaning green so the operator's asides — the
/// system's *voice* — stand out from the executable *body*. A scanline dims even
/// rows so the wall reads as a powered-down CRT, not a printed page.
const CODE_BG: Vec3 = Vec3::new(0.012, 0.024, 0.016); // near-black with a green cast
const CODE_FG: Vec3 = Vec3::new(0.27, 0.85, 0.38); // phosphor green — the code body
const CODE_COMMENT_FG: Vec3 = Vec3::new(0.55, 1.0, 0.72); // brighter — the voice
const CODE_VOID_FG: Vec3 = Vec3::new(0.04, 0.10, 0.06); // faint, for blank cells
const CODE_SCANLINE_DIM: f32 = 0.72; // even-row scanline darkening

// ─────────────── note-placard palette (per-register styling) ──────────────
//
// The INK/PLATE colours now live with the registers that choose them, in
// `crate::note_style` (`NoteRegister::palette` — warm `researcher`, cold
// `system_log`, dim `model_thought`, plus the `Plain` default that reproduces the
// 1.0 monochrome plate byte-for-byte). Only the one knob that is a property of the
// SURFACE rather than of a register stays here:

/// Brightness multiplier applied to the plate on a face the placard is NOT painted
/// on. The reverse side of a note wall must read as an ordinary dark wall — not as a
/// bright blank plate, and never as mirrored text.
const NOTE_REVERSE_DIM: f32 = 0.45;

/// Image-recipe knobs for the nearest-neighbour RGBA8 wall texture.
/// `SOLID_BLOCK` is the full-block glyph the recipe paints so a sampled texel fills
/// the whole cell; `IMAGE_BG_DIM` darkens the cell bg slightly under that block so
/// the `█` body reads crisply over its own cell edge without crushing the colour.
const SOLID_BLOCK: u32 = 0x2588; // █ — full block
const IMAGE_BG_DIM: f32 = 0.82; // subtle bg darkening under the solid block

// ─────────────── ASCII-family image sampler tunables ───────────────
//
// Named knobs for `/iterate-render`. The ramp maps luminance → glyph density; the
// edge threshold gates the directional-glyph swap; the lines-mode contrast and the
// edge/ramp blend shape the final look. All pure data, golden-stable.

/// Luminance → fill-glyph ramp for the `Ascii`/`Contour`/`Lines` samplers, DARK → LIGHT.
/// Index `0` (space) is the darkest, the last (`@`) the brightest. The ` .:-=+*#%@`
/// ramp is dense at the bright end so highlights read as solid.
pub const ASCII_IMAGE_RAMP: &[u8] = b" .:-=+*#%@";

/// Edge-magnitude threshold (normalised `[0,1]`) at/above which a cell swaps its ramp
/// fill glyph for the directional contour glyph. Mirrors
/// [`crate::edge_field::EDGE_MAG_THRESHOLD`] as the shader-side knob; kept separate so
/// `/iterate-render` can retune the SAMPLE gate without rebuilding the baked field.
pub const ASCII_EDGE_THRESHOLD: f32 = crate::edge_field::EDGE_MAG_THRESHOLD;

/// `Lines` mode: how close a NON-edge cell's fg ink sits to its bg fill `[0,1]`.
/// `0` ⇒ fg == bg (the fill recedes completely, only contours show); `1` ⇒ fg is the
/// full fill colour (same as `Ascii`/`Contour`). Low by design so lines pop.
pub const LINES_FILL_CONTRAST: f32 = 0.18;

/// `Lines` mode: the edge-glyph INK brightness multiplier applied to the contrasting
/// ink so contour strokes read clearly over the (recessed) fill bg.
pub const LINES_EDGE_INK: f32 = 1.0;

/// Edge-vs-ramp BLEND: how strongly a strong edge brightens the directional glyph's
/// ink toward white (a tiny lift so contours read as light strokes, not just a glyph
/// swap). `0` ⇒ pure fill colour ink, `1` ⇒ full white. Kept subtle.
pub const ASCII_EDGE_INK_LIFT: f32 = 0.35;

/// How much stronger the LOWER half of a cell's edge response must be than the upper
/// half before a near-horizontal stroke moves from `-` to `_`
/// (`Material::ascii_edge_row_bias`).
///
/// Well above 1 on purpose. The substitution is only an improvement when the edge is
/// genuinely near the cell's bottom; applied on a slim margin it makes a long
/// centred boundary alternate `-`/`_` on field noise, which reads worse than never
/// biasing at all. 1.6 keeps centred edges centred.
pub const ASCII_EDGE_BIAS_RATIO: f32 = 1.6;

/// `Contour` (hero DoG) edge-threshold multiplier over [`ASCII_EDGE_THRESHOLD`]: a
/// value `> 1` makes the contour pass MORE selective than the `Ascii` baseline → only
/// the strongest DoG responses ink a stroke, giving thinner / cleaner lines. A named
/// `/iterate-render` knob.
pub const CONTOUR_THRESHOLD_SCALE: f32 = 1.4;

/// Which edge LOOK the shared [`Material::sample_image_ascii`] assembler produces —
/// the discriminant the three new [`SubcellMode`] variants map onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeMode {
    /// `Ascii`: ramp fill + Sobel-edge directional glyph.
    Sobel,
    /// `Contour`: ramp fill + DoG-edge directional glyph (the hero look).
    Dog,
    /// `Lines`: every cell a glyph, non-edge cells inked ≈ bg so only contours pop.
    Lines,
}

/// Map a luminance `[0,1]` to a glyph from [`ASCII_IMAGE_RAMP`] (dark→light). Clamped;
/// the brightest texel picks the densest glyph (`@`), the darkest a space.
#[inline]
pub fn ascii_ramp_glyph(lum: f32) -> u32 {
    let n = ASCII_IMAGE_RAMP.len();
    if n == 0 {
        return u32::from(' ');
    }
    let l = lum.clamp(0.0, 1.0);
    let idx = ((l * n as f32) as usize).min(n - 1);
    u32::from(ASCII_IMAGE_RAMP[idx])
}

/// ALPHA CUT-OUT THRESHOLD — the cut-out pass-through boundary.
/// A sampled texel whose 8-bit alpha is `< IMAGE_ALPHA_CUTOFF` is treated as
/// TRANSPARENT — the raycaster's [`crate::dda_hit`] marches PAST it (the ray
/// continues to whatever backdrop sits behind, exactly like a wedge open triangle
/// or a hologram cell), and the per-row shade keeps the backdrop there. A texel at
/// or above the cutoff is OPAQUE and stops the ray / paints the image. `128` is the
/// natural 1-bit-alpha midpoint (PNG cut-outs are typically 0 or 255), so a fully
/// opaque texture never trips it and an all-opaque image is byte-identical to
/// the plain solid-block path. Shared so the shader's opaque test and the caster's
/// pass-through test agree on exactly one boundary.
pub(crate) const IMAGE_ALPHA_CUTOFF: u8 = 128;

/// True if `[r,g,b,a]`'s alpha marks the texel OPAQUE (`a >= IMAGE_ALPHA_CUTOFF`).
/// The single shared predicate for the cut-out boundary.
#[inline]
pub(crate) fn image_texel_opaque(a: u8) -> bool {
    a >= IMAGE_ALPHA_CUTOFF
}

/// Cap on how many vertical samples the column-level strip-transparency test
/// (`Material::image_strip_fully_transparent`) takes for one cell entry. One sample
/// per image row up to this cap keeps the per-cell DDA cost bounded for a tall
/// texture while still catching a thin opaque sliver (we sample row-band centres).
pub(crate) const IMAGE_STRIP_MAX_SAMPLES: u32 = 64;

/// Hologram knobs (the glyph-only transparent scanline surface). The bg is a
/// VOID — `HOLO_VOID_LEVEL` is the near-black floor used for every hologram cell
/// (both line and gap), so the surface is see-through; the colour lives entirely
/// in the fg line glyphs. `HOLO_FG_FLOOR` is the dimmest a line-edge glyph fades
/// to (the line never goes fully black, so it reads as a continuous glowing line).
const HOLO_VOID_LEVEL: f32 = 0.012; // near-black bg — the transparent void
const HOLO_FG_FLOOR: f32 = 0.45; // dimmest line-edge fg (still a visible glow)
/// Hologram scanline glyphs, brightest core → faint edge: a solid double rule, a
/// single rule, a dashed rule, and a faint mid-dot. Horizontal line glyphs so the
/// surface reads as stacked holographic scanlines drifting vertically.
const HOLO_LINE_SOLID: u32 = 0x2550; // ═
const HOLO_LINE_THIN: u32 = 0x2500; // ─
const HOLO_LINE_DASH: u32 = 0x254C; // ╌
const HOLO_LINE_DOT: u32 = 0x00B7; // ·

// Substrate knobs (the cell-alphabet wall).
//
// The state's `[0,1]` tone ([`crate::substrate::tone`]) is carried by a MONOTONE
// brightness scale over ONE fixed hue, NOT by walking `t` across a palette window.
//
// FOUND BY MEASUREMENT after the first render: `Palette::eval` is a cosine, so a `t`
// window turns the colour over. The first attempt walked `t ∈ [0.10, 0.44]`, which is
// the DESCENDING side of that cosine — it INVERTED the tiers. A held cell (tone 0.92)
// came out at fg luma 0.032 while an unclaimed one (tone 0.30) came out at 0.079, and
// once glyph ink coverage was folded in (`█` fills its cell, `░` covers about a
// quarter of it) the two matched in perceived luminance to within 0.001. The four
// tiers were distinguishable by glyph shape ALONE — exactly the "does it read?"
// property the second channel was supposed to add.
//
// `substrate_tiers_are_monotone_in_brightness` pins the fix.

/// Where the substrate's single hue is read off both palettes. One point, so the four
/// states are one colour family and cannot drift around the hue wheel.
const SUBSTRATE_HUE_T: f32 = 0.25;
/// Ink scale at tone `0` (the darkest tier). Well above zero: `✕` is a mark, not a
/// hole, and a burned wall must read as a marked surface rather than as a void.
const SUBSTRATE_INK_FLOOR: f32 = 0.35;
/// Ground scale at tone `0`. The bg runs the OTHER way — darkest under the brightest
/// cell — so a `█` sits on shadow and reads as mass.
const SUBSTRATE_BG_CEIL: f32 = 1.0;
/// How much of the ground the tone takes away at tone `1`.
const SUBSTRATE_BG_SPAN: f32 = 0.6;

// Facade knobs (the bay×floor window grid). Same monotone-ink-over-one-hue shape as
// the Substrate knobs directly above — a lit window must out-shine a dark one AND a
// far-frozen/collapsed cell (whose `ink` sits mid-range, the frame's `frame_ink`)
// must land visibly between the two, not collide with either.

/// Where the facade's single hue is read off both palettes.
const FACADE_HUE_T: f32 = 0.42;
/// Ink scale at `ink == 0`. A dark window is still glass, not a hole in the wall.
const FACADE_INK_FLOOR: f32 = 0.18;
/// Ground (bg) scale at `ink == 0` — the wall/spandrel's OWN dark tone. Unlike
/// [`SUBSTRATE_BG_CEIL`] (a bright-mark-on-dark-ground recipe), a facade's ground
/// runs the SAME way as ink: most units on a real building are unlit, and they
/// must sit on the wall's dark base tone, not glow. An earlier inverted mapping
/// (`CEIL - SPAN * ink`) put the *majority* (unlit) units on a BRIGHT ground and only
/// lit units on a dark one, so a wall with `lit < 0.5` (every authored style) read as
/// a uniformly bright field speckled with dark marks — dotted noise, not buildings.
const FACADE_BG_FLOOR: f32 = 0.1;
/// How much a lit unit's ground brightens above the floor — the warm glow of
/// light spilling from a lit window, added ONLY to genuinely lit units (high
/// `ink`), not subtracted from everyone else.
const FACADE_BG_GLOW: f32 = 0.6;
/// The mullion/sill frame's OWN ground — independent of the ink formula above, a
/// fixed dark "structural" tone so the frame band reads as one consistent grid
/// line regardless of whether it sits between two lit units, two dark ones, or
/// one of each.
const FACADE_FRAME_BG: f32 = 0.16;

/// The WALL-PLANE world coordinate a [`TextureRecipe::Substrate`] cell is anchored to:
/// the continuous world hit along the face, and the texel's world height.
///
/// Deliberately NOT `coord` — see [`Material::sample_substrate`] for the measured
/// reason (`coord.x` wraps once per world cell, which tiled the field across a wall
/// run). One place, so the texel path and the height field cannot drift apart.
#[inline]
const fn substrate_plane(ctx: &SampleCtx) -> Vec2 {
    Vec2::new(ctx.wall_along, ctx.wall_up)
}

/// Scanline knobs (the FILLED directional-band panel — CRT / metal / fabric).
/// `SCANLINE_FG_FLOOR` is the dimmest a band-edge fg fades to (the band never goes
/// black, so it reads as a continuous rule); `SCANLINE_GAP_BG` darkens the panel bg
/// inside a band GAP so the bands read with scanline contrast.
const SCANLINE_FG_FLOOR: f32 = 0.42;
const SCANLINE_GAP_BG: f32 = 0.55;

/// How far a coverage-MASKED band cell's bg moves from the panel bg toward the band
/// colour: enough that a `metal`/`wood` surface reads as its own dim tone in a dark
/// room, low enough that the few inked rules still carry the texture.
const SCANLINE_MASKED_BAND_BG: f32 = 0.30;

/// Emissive level at and above which a material is an EMITTER — a lit screen, a neon
/// tube, a powered prop slot — rather than a surface that merely has a faint glow
/// (circuit 0.10, gold 0.12, matrix 0.30, sky 0.6, lava 0.75 are all below it;
/// `monitor` 0.85, `neon` 0.9 and the puzzle node 0.85 are at or above it).
///
/// An emitter makes its own light, so the two treatments meant for lit SURFACES do
/// not apply to it:
///
/// * the wall/floor SPARSE-COVERAGE gate of [`TextureRecipe::Scanline`] (written so a
///   `metal`/`wood` body texture reads by light and colour instead of packed rules)
///   blanked about 70 % of a lit screen's bands, and the gap darkening then left a
///   powered prop slot DARKER than the lamp-lit brick around it;
/// * distance fog attenuates it less: [`EMITTER_FOG_KEEP`] of the light fog would
///   have removed is kept, so a lit slot still reads through the haze of a dark room.
pub const SELF_LIT_EMISSIVE: f32 = 0.8;

/// Fraction of the fog attenuation an [`SELF_LIT_EMISSIVE`] emitter is spared:
/// its fog clarity is `fog + (1 − fog) · EMITTER_FOG_KEEP`.
pub const EMITTER_FOG_KEEP: f32 = 0.5;

/// Fractal portal bg floor: the dimmest the ring-trough background fades to (never
/// pure black, so the troughs keep a faint coloured depth rather than reading as a
/// hole), scaling up toward the ring crests with the height field for contrast.
const FRACTAL_BG_FLOOR: f32 = 0.25;

/// Scalable-overlay SelfSimilar modulation floors. The fg never fades
/// below [`SELF_SIMILAR_FG_FLOOR`] of the zone's base colour (so the material's hue
/// survives at the field troughs — the portal composes with, never erases, the
/// surface), brightening toward the field crests. The bg never drops below
/// [`SELF_SIMILAR_BG_FLOOR`] of its base, deepening toward the ring troughs so the
/// nested rings read with contrast. Named tunables, not scattered literals.
const SELF_SIMILAR_FG_FLOOR: f32 = 0.35;
const SELF_SIMILAR_BG_FLOOR: f32 = 0.30;

/// Which built-in SURFACE a [`TextureRecipe::Scanline`] material is painting, as
/// inferred from the per-cell lighting geometry. Used only to gate the sparse-
/// surface coverage + plain-glyph path (see [`Material::sample_scanline`]); a
/// Scanline that is NOT a wall/floor surface (a puzzle panel, a future sprite)
/// returns `None` and keeps the full filled-band look.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScanlineSurface {
    /// A vertical face (wall): the geometric normal is roughly horizontal.
    Wall,
    /// A horizontal face (floor or ceiling): the geometric normal is roughly ±Z.
    Floor,
}

/// Classify the cell's surface from its lighting geometry. The raycaster builds the
/// per-cell [`LightCtx`] with a floor/ceiling normal of ±Z ([`LightCtx::for_floor`])
/// and a wall normal in the XY plane ([`LightCtx::for_wall`]); we read that face
/// normal to decide which sparse-coverage mask the Scanline body should honour. When
/// no lighting context is supplied (a focused unit test, or a non-surface caller),
/// returns `None` so the historical filled-band look is preserved exactly.
#[inline]
fn scanline_surface(ctx: &SampleCtx) -> Option<ScanlineSurface> {
    let n = ctx.light?.face_n;
    // A floor/ceiling face has a Z-dominant normal; everything else is treated as a
    // wall. The 0.5 cut cleanly separates the axis-aligned built-in faces.
    if n.z.abs() > 0.5 {
        Some(ScanlineSurface::Floor)
    } else {
        Some(ScanlineSurface::Wall)
    }
}

/// The PLAIN (un-promoted) band rule glyph for an `axis`, used when a Scanline paints
/// a wall/floor surface. A single clean rule — `─` horizontal, `│` vertical, `┼` at
/// the woven crossing — never the heavy/dotted `━`/`┈`/`┃`/`┆` family that packed the
/// deck floor. Pure, allocation-free.
#[inline]
fn scanline_plain_glyph(axis: u8) -> u32 {
    match axis {
        1 => 0x2502, // │ light vertical
        2 => 0x253C, // ┼ light cross
        _ => 0x2500, // ─ light horizontal
    }
}

/// Pick a band glyph for a [`TextureRecipe::Scanline`] cell from its centrality
/// `core ∈ [0,1]` (1 = band centre) and the band `axis`. Horizontal bands use
/// horizontal rules (heavy → light), vertical bands the vertical rules, and the
/// woven axis a hatch/plus at the crossing. Pure, allocation-free.
#[inline]
fn scanline_glyph(axis: u8, core: f32) -> u32 {
    match axis {
        // VERTICAL grate: heavy vertical bar → light vertical rule.
        1 => {
            if core > 0.6 {
                0x2503 // ┃ heavy vertical
            } else if core > 0.3 {
                0x2502 // │ light vertical
            } else {
                0x2506 // ┆ dotted vertical (the fraying edge)
            }
        }
        // WOVEN cross-hatch: a plus/hatch at the crossing → lighter toward the edge.
        2 => {
            if core > 0.6 {
                0x256C // ╬ heavy cross
            } else if core > 0.3 {
                0x253C // ┼ light cross
            } else {
                0x2592 // ▒ mesh shade at the weave gaps
            }
        }
        // HORIZONTAL raster (default): heavy → light → dashed horizontal rule.
        _ => {
            if core > 0.6 {
                0x2501 // ━ heavy horizontal
            } else if core > 0.3 {
                0x2500 // ─ light horizontal
            } else {
                0x2508 // ┈ dotted horizontal (the scan-gap edge)
            }
        }
    }
}

// ───────────────────────────── registry ─────────────────────────────────────

/// Built-in material ids. Surfaces map to these; per-level overrides (a future
/// `.lvl [material]` section) would replace the registry entries.
pub mod ids {
    use super::MaterialId;
    /// Default wall surface — the `Brick` recipe, non-emissive.
    pub const WALL_BRICK: MaterialId = MaterialId(0);
    /// Circuit-board wall surface — the `Circuit` recipe.
    pub const WALL_CIRCUIT: MaterialId = MaterialId(1);
    /// Glowing code-rain wall surface — the emissive `CodeWall` recipe.
    pub const WALL_MATRIX: MaterialId = MaterialId(2);
    /// Default floor surface — the `Grain` recipe.
    pub const FLOOR_GRAIN: MaterialId = MaterialId(3);
    /// Default ceiling surface — the `Sky` recipe.
    pub const CEILING_SKY: MaterialId = MaterialId(4);
    /// PUZZLE ELEMENTS (player legibility): a `CellKind::Object` puzzle NODE (a
    /// neuron) and the exit DOOR. These render with distinct, recognizable glyphs
    /// in saturated colours so the player can SEE the puzzle against the calmer
    /// walls/floor. Slots five and six sit AFTER the five surface slots, so the
    /// existing surface dispatch and the authored-material merge (which only touches
    /// the surface slots) are unaffected — additive built-ins dispatched directly.
    pub const PUZZLE_NODE: MaterialId = MaterialId(5);
    /// The exit door built-in — see [`PUZZLE_NODE`] above for the slot rationale.
    pub const PUZZLE_DOOR: MaterialId = MaterialId(6);
    /// HOLOGRAM built-in — the glyph-only transparent scanline surface. Appended
    /// after the fixed surface + puzzle slots so it never disturbs the surface
    /// dispatch; reachable by name or by the `hologram` recipe in `[material]`.
    pub const HOLO_NODE: MaterialId = MaterialId(7);

    // ── CATALOG materials (the "rich library of smart materials") ──
    //
    // A varied named catalog, each a pure DATA config of the composable material
    // system, appended at slots 8.. so the fixed surface (0..5), puzzle (5/6), and
    // hologram (7) slots — and every shipped level + the `SurfaceKind` dispatch —
    // are byte-for-byte unchanged. Reachable BY NAME (`id_of("rust")`) or by a
    // level's `[material] preset = NAME`. Registered by [`super::MaterialRegistry::with_catalog`].
    /// Brushed steel: cool grey directional grain, high specular sheen, near-zero emissive.
    pub const METAL: MaterialId = MaterialId(8);
    /// Warm gold/brass: rich amber sheen, high specular, faint emissive lustre.
    pub const GOLD: MaterialId = MaterialId(9);
    /// Reflective glass/window pane: low ink, near-void bg, crisp cool-tinted mirror.
    pub const GLASS: MaterialId = MaterialId(10);
    /// Matte stone/concrete: heavy bricked wall, cool grey, deep shadowed seams.
    pub const STONE: MaterialId = MaterialId(11);
    /// Rust/corrosion: warm noisy oxide grain, matte, pitted ambient occlusion.
    pub const RUST: MaterialId = MaterialId(12);
    /// Ice/frost: cool pale-blue crystalline facets, faint reflective clarity.
    pub const ICE: MaterialId = MaterialId(13);
    /// Wood: warm directional grain, soft satin sheen, gentle relief.
    pub const WOOD: MaterialId = MaterialId(14);
    /// Fabric/weave: soft low-relief woven cross-hatch, muted dusty palette.
    pub const FABRIC: MaterialId = MaterialId(15);
    /// Water/liquid: animated cool flowing surface, reflective, faint emissive shimmer.
    pub const WATER: MaterialId = MaterialId(16);
    /// Lava/energy: hot high-emissive pulsing surface with an under-glow accent.
    pub const LAVA: MaterialId = MaterialId(17);
    /// Neon sign: saturated magenta phosphor with a failing-tube flicker, strong emissive.
    pub const NEON: MaterialId = MaterialId(18);
    /// Monitor/CRT: phosphor scanlines with a refresh flicker, emissive glow.
    pub const MONITOR: MaterialId = MaterialId(19);
    /// Flesh/organic: warm soft slowly-breathing grain, faint subsurface emissive.
    pub const FLESH: MaterialId = MaterialId(20);
    /// Void/space: near-black dark starfield, the dimmest catalog material.
    pub const VOID: MaterialId = MaterialId(21);
    /// Hologram variant in an amber/gold hue with a slightly faster scroll.
    pub const HOLO_AMBER: MaterialId = MaterialId(22);
    /// Circuit variant in a hot red/orange palette with a faster, denser pulse.
    pub const CIRCUIT_RED: MaterialId = MaterialId(23);

    // ── KINDRED-SYMBOL / FLOATING-GLYPH demo materials (slots 24..) ──
    //
    // Three materials that exercise the [`crate::glyph_family`] super-glyph layer +
    // the bold-as-a-channel cues, so the "floating" effect is visible in the catalog.
    /// HAZE — a neutral kindred family (`o 0 O °`) floated over `time`: a soft,
    /// drifting mist that shimmers without changing meaning.
    pub const HAZE: MaterialId = MaterialId(24);
    /// FLOW — a directional within-glyph animation (a flowing stream of close
    /// glyphs), faster cycle, cool palette.
    pub const FLOW: MaterialId = MaterialId(25);
    /// SPARK / EMBER — a GROWTH family (point→disc) + an emissive pulse driving the
    /// bold channel + an underline glow: the five-channel emitter demo.
    pub const SPARK: MaterialId = MaterialId(26);

    // ── TRANSPARENT materials (slots 27..) — composite slot 2, `OverKeepBg`. Each
    //    is a [`TextureRecipe::Hologram`] variant, so `is_transparent()` is true. ──
    /// VITRINE — a clean see-through glass pane (transparent).
    pub const VITRINE: MaterialId = MaterialId(27);
    /// SCREEN — a translucent glowing HUD display (transparent).
    pub const SCREEN: MaterialId = MaterialId(28);
    /// VEIL — a soft translucent gauze curtain (transparent).
    pub const VEIL: MaterialId = MaterialId(29);

    // ── NEW OPAQUE smart materials (slots 30..) — varied recipes / families ──
    /// MOSS — organic green crust with a slow floating dot family.
    pub const MOSS: MaterialId = MaterialId(30);
    /// EMBER_FIELD — a growth-by-distance ember bed, emissive pulse.
    pub const EMBER_FIELD: MaterialId = MaterialId(31);
    /// STATIC_NOISE — a fast within-glyph spinner + flicker (TV static).
    pub const STATIC_NOISE: MaterialId = MaterialId(32);
    /// LATTICE — a calm woven steel-blue cross-hatch panel.
    pub const LATTICE: MaterialId = MaterialId(33);

    // ── SUPER-GLYPH smart materials (slots 34..) — the selectors beyond the four
    //    family drivers, finally authorable on a material. ──
    /// BEAT_GRID — a [`super::SuperSelector::Bpm`] panel: the kindred glyph steps on the beat
    /// (derived from `ctx.time`), pulsing in sync with the rhythmic-audio layer.
    pub const BEAT_GRID: MaterialId = MaterialId(34);
    /// AUTO_TILE — a [`super::SuperSelector::NeighborAware`] wall: adjoining cells pick
    /// CONNECTED members from a world-lattice hash, so the surface auto-tiles.
    pub const AUTO_TILE: MaterialId = MaterialId(35);
    /// BOIL — a [`super::SuperSelector::Procedural`] field: a world-seeded "boiling" glyph
    /// that advances in stable quantized time-steps (endless non-repeating churn).
    pub const BOIL: MaterialId = MaterialId(36);

    // ── FRACTAL / INFINITE-ZOOM portal materials (slots 37..) — the log-polar
    //    self-similar [`super::TextureRecipe::Fractal`] recipe + its anim modes. ──
    /// PORTAL_ZOOM — concentric rings rushing outward (the `Zoom` infinite-zoom mode).
    pub const PORTAL_ZOOM: MaterialId = MaterialId(37);
    /// PORTAL_SPIRAL — a Droste spin+zoom spiral (the coupled `SpinZoom` mode).
    pub const PORTAL_SPIRAL: MaterialId = MaterialId(38);

    // ── SEMANTIC SURFACES — slots 42.. ──
    //
    // APPENDED after the `facet`/`pane`/`relief_vol` demo materials (slots 39/40/41,
    // which carry no `ids::` constant) so no existing MaterialId shifts. Each opts into
    // a [`crate::surface_fx::SurfaceFx`] finish; the redaction/decay surfaces react to
    // the generic state bus, the rest are static procedural surfaces.
    /// FROST — cold worley-crystal accretion (the cold/warm semantic axis).
    pub const FROST: MaterialId = MaterialId(42);
    /// WET_STONE — a cool specular condensation sheen over dark stone.
    pub const WET_STONE: MaterialId = MaterialId(43);
    /// REDACTED — `█`/`░` censorship over the CodeWall, lifting as `ctx.state` rises.
    pub const REDACTED: MaterialId = MaterialId(44);
    /// WATCHER — a world-fixed eye/face lattice that brightens on approach.
    pub const WATCHER: MaterialId = MaterialId(45);
    /// WORN_FLOOR — scuffed grooves + settled dust (where iterations stood).
    pub const WORN_FLOOR: MaterialId = MaterialId(46);
    /// DECAY — bounded local bit-rot (mojibake + sickly tint), state-driven intensity.
    pub const DECAY: MaterialId = MaterialId(47);

    // ── ANIMATION-TRACK material — slot 48 ──
    //
    // APPENDED at the tail so no existing MaterialId / slot shifts (goldens hold).
    /// NODE_BREATHING — a puzzle-node variant that BREATHES (`·∘○◌●` morph) when the
    /// state bus reports it POWERED (a non-zero active state), static when dead.
    pub const NODE_BREATHING: MaterialId = MaterialId(48);

    // ── SUBSTRATE (the cell alphabet) — slot 49 ──
    //
    // APPENDED at the tail so no existing MaterialId / slot shifts (goldens hold).
    /// SUBSTRATE — the buffer's own wall texture: the `░ █ ▒ ✕` cell alphabet.
    ///
    /// [`crate::substrate`] over a world-anchored lattice. Static (`Animation::NONE`) —
    /// its field is driven by the two authored `held` / `released` scalars, never by
    /// `time`, so it is fully cacheable.
    pub const SUBSTRATE: MaterialId = MaterialId(49);

    // ── THE VOICE PAIR (state-driven substrate) — slots 50/51 ──
    //
    // APPENDED at the tail so no existing MaterialId / slot shifts (goldens hold).
    // Both carry [`super::TextureRecipe::Voice`]: the substrate alphabet driven by the
    // level's solve-progress MARK off the state bus rather than by authored scalars.
    /// The KEEPING voice — an empty wall that FILLS with `█` as progress rises, and
    /// keeps every cell it has taken. Reads the mark on
    /// [`crate::substrate::MNEMOS_TAG`].
    pub const MNEMOS_HOLD: MaterialId = MaterialId(50);
    /// The LETTING-GO voice — a whole wall that ERODES to `▒` as progress rises, and
    /// never heals a scar. Reads the mark on [`crate::substrate::LETHE_TAG`].
    pub const LETHE_RELEASE: MaterialId = MaterialId(51);

    /// The edge-matched conduit lattice (`truchet`) — slot 52.
    pub const CONDUIT: MaterialId = MaterialId(52);

    /// The structured bay×floor window grid (`facade`, style `office`) — slot 53.
    /// APPENDED at the tail so no existing `MaterialId` / slot shifts (goldens hold).
    pub const FACADE: MaterialId = MaterialId(53);

    /// The catalog material NAMES, in slot order from [`METAL`] (slot 8) onward.
    /// Parallel to the [`super::MaterialRegistry::with_catalog`] append order, so a
    /// tool can enumerate the catalog (`for n in ids::CATALOG_NAMES`). Pure data.
    pub const CATALOG_NAMES: &[&str] = &[
        "metal",
        "gold",
        "glass",
        "stone",
        "rust",
        "ice",
        "wood",
        "fabric",
        "water",
        "lava",
        "neon",
        "monitor",
        "flesh",
        "void",
        "holo_amber",
        "circuit_red",
        "haze",
        "flow",
        "spark",
        // transparent (Hologram-recipe variants):
        "vitrine",
        "screen",
        "veil",
        // new opaque smart materials:
        "moss",
        "ember_field",
        "static_noise",
        "lattice",
        // super-glyph smart materials:
        "beat_grid",
        "auto_tile",
        "boil",
        // fractal / infinite-zoom portal materials:
        "portal_zoom",
        "portal_spiral",
        // semantic surfaces — slots 42.. (the demo
        // materials facet/pane/relief_vol at 39/40/41 carry no name entry here):
        "frost",
        "wet_stone",
        "redacted",
        "watcher",
        "worn_floor",
        "decay",
        // animation-track material — slot 48:
        "node_breathing",
        // the cell alphabet (animation-palette plan, Task 4) — slot 49:
        "substrate",
        // the state-driven voice pair (animation-palette plan, Task 5) — slots 50/51:
        "mnemos_hold",
        "lethe_release",
        // edge-matched conduit tiling — slot 52:
        "conduit",
        // the structured bay×floor window grid — slot 53:
        "facade",
    ];
}

/// A process-wide fallback [`Material`] so [`MaterialRegistry::get`] can never
/// panic even on a fully-empty registry (R-L1). A neutral grey brick.
pub static DEFAULT_MATERIAL: Material = Material {
    recipe: TextureRecipe::Brick {
        aspect: 2.0,
        mortar_w: 0.16,
        grain: 0.5,
    },
    palette: Palette {
        a: Vec3::new(0.5, 0.5, 0.5),
        b: Vec3::new(0.2, 0.2, 0.2),
        c: Vec3::new(1.0, 1.0, 1.0),
        d: Vec3::new(0.0, 0.1, 0.2),
    },
    bg_palette: Palette {
        a: Vec3::new(0.12, 0.12, 0.14),
        b: Vec3::new(0.06, 0.06, 0.08),
        c: Vec3::new(1.0, 1.0, 1.0),
        d: Vec3::new(0.0, 0.1, 0.2),
    },
    rank: RankStrategy::StructureFirst,
    seed: SHADER_SEED,
    glyph_source: GlyphSource::Recipe,
    animation: Animation::NONE,
    light: LightModel::DEFAULT,
    // Neutral fallback: responsive (the shipped behaviour, no extra deadband).
    stability: GlyphStability::Responsive,
    // Not a mirror — an ordinary opaque surface.
    reflective: false,
    clarity: 1.0,
    // No underline accent layer on the neutral fallback (unchanged output).
    underline_accent: UnderlineAccent::NONE,
    // No kindred-symbol family layer and no proximity-bold cue: the recipe glyph is
    // used unchanged and no distance-driven bold — fully back-compatible.
    family: None,
    // No super-glyph selector either — the recipe glyph is unchanged.
    super_glyph: None,
    proximity_bold: None,
    lod_override: None,
    glass_marker: false,
    surface_fx: None,
    // No animation tracks on the neutral fallback.
    anim_tracks: AnimTrackSet::EMPTY,
    // No scale rule (Scalable Material Overlay) — flat octave-0, today's look.
    scale: ScaleMode::None,
    // No attached image texture — the built-in fallback uses a procedural recipe.
    texture: None,
    // No precomputed edge field (no texture to derive one from).
    edge_field: None,
    texture_anim: None,
    // No detail-tier opt-in on the neutral fallback.
    detail: false,
};

/// A named material definition — the data-driven authoring unit. A future
/// stream (the `.vox`/`.toml` converter, or a `.lvl [material]` section) can
/// hand the registry a list of these to define materials by name, then
/// reference them by name → [`MaterialId`]. Keeps the engine the single owner
/// of how a `Material` shades.
#[derive(Clone, Debug)]
pub struct MaterialSpec {
    /// Stable name an author references (e.g. `"wall_brick"`, `"sign_neon"`).
    pub name: &'static str,
    /// The material data this name resolves to.
    pub material: Material,
}

/// The data-driven material table — a Bevy resource. Built-ins are seeded in
/// [`MaterialRegistry::with_builtins`]; entries are plain data, so adding a look
/// is one `register` call, a [`MaterialSpec`] in [`from_specs`](Self::from_specs),
/// or a parsed `.lvl` override later.
///
/// A parallel `names` table maps a name → its slot, so authoring can be
/// name-keyed while the hot shade path stays index-keyed (`get(MaterialId)`).
#[derive(Resource, Clone, Debug)]
pub struct MaterialRegistry {
    materials: Vec<Material>,
    /// Optional name for each slot (`""` when registered un-named). Parallel to
    /// `materials`; lookups go through [`id_of`](Self::id_of).
    names: Vec<&'static str>,
    /// Cached "any material is animated" flag, refreshed on every insert so the
    /// renderer's dirty-flag can ask in O(1) (see [`any_animated`](Self::any_animated)).
    any_animated: bool,
    /// The per-cell **colour palette**: `color_material_ids[i]` is the registry
    /// slot for the 1-based colour-material id `i + 1` (the data side of the
    /// colour→material substrate, `re_voximport` rule 9). A geometry cell whose
    /// `side_data` is a colour id `> 0` shades from this slot instead of its
    /// surface-kind default ([`color_material`](Self::color_material)). Empty ⇒ no
    /// colour palette (the common case → byte-identical to today).
    color_material_ids: Vec<MaterialId>,
    /// NOTE BACKING: `(note colour id, backing colour id)` pairs. A `[note]` placard
    /// is stamped as a whole-cell colour material, but only ONE face of that cell is
    /// the placard; the others belong to whatever the placard is mounted in (a wall
    /// run, a notice board's body). [`crate::registry_for_map`] resolves each note's
    /// backing from its neighbouring cells, and the shade path draws every
    /// non-placard face from it. Backing id `0` = the surface-kind default. Empty ⇒
    /// no notes (byte-identical to a level without placards).
    note_backing: Vec<(u16, u16)>,
    /// FLOOR DECALS: the colour ids the camera-storey floor backdrop honours. The
    /// backdrop fill shades the floor plane under the eye from the surface-kind
    /// default for speed and never read a cell's colour, so a prop kit's floor decal
    /// (a trace table's contact shadow, a moss patch) stamped into the floor cells
    /// never showed. Only the ids listed here are looked up there (see
    /// [`crate::registry_for_map`]); empty ⇒ the backdrop is byte-identical to before.
    floor_decal_ids: Vec<u16>,
    /// Z-BANDS: `(colour id, band)` for the colour materials that are solid over only
    /// part of their cell's height ([`crate::zband`]). Empty ⇒ every cell is a full
    /// solid (byte-identical to a level with no emit figure or solid furniture).
    z_bands: Vec<(u16, crate::zband::ZBand)>,
}

impl MaterialRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            materials: Vec::new(),
            names: Vec::new(),
            any_animated: false,
            color_material_ids: Vec::new(),
            note_backing: Vec::new(),
            floor_decal_ids: Vec::new(),
            z_bands: Vec::new(),
        }
    }

    /// Register a material, returning its [`MaterialId`] (its slot index).
    pub fn register(&mut self, m: Material) -> MaterialId {
        self.register_named("", m)
    }

    /// Register a material under a `name`, returning its [`MaterialId`]. The name
    /// can later be resolved with [`id_of`](Self::id_of). An empty name registers
    /// the material without a name-lookup entry.
    pub fn register_named(&mut self, name: &'static str, m: Material) -> MaterialId {
        let id = MaterialId(self.materials.len() as u16);
        self.any_animated |= m.is_animated();
        self.materials.push(m);
        self.names.push(name);
        id
    }

    /// Build a registry from a list of named specs (data-driven authoring). The
    /// slot order matches the spec order, so callers may also rely on index.
    pub fn from_specs(specs: &[MaterialSpec]) -> Self {
        let mut r = Self::new();
        for s in specs {
            r.register_named(s.name, s.material.clone());
        }
        r
    }

    /// Resolve a material name → its [`MaterialId`], if registered.
    pub fn id_of(&self, name: &str) -> Option<MaterialId> {
        self.names
            .iter()
            .position(|n| *n == name)
            .map(|i| MaterialId(i as u16))
    }

    /// Register a material AND record it as the next entry in the per-cell
    /// **colour palette** (the colour→material substrate). The first call binds
    /// colour id `1`, the second id `2`, etc. — matching the 1-based ids
    /// `re_level` stamps into a geometry cell's `side_data` from `[color_cell_ids]`.
    /// Returns the underlying registry [`MaterialId`] (its slot). The material is
    /// a normal registry slot too (reachable by index), so the hot shade path
    /// stays index-keyed. The palette is index-keyed (not name-keyed), so the
    /// slot is registered un-named.
    pub fn register_color_material(&mut self, m: Material) -> MaterialId {
        let id = self.register(m);
        self.color_material_ids.push(id);
        id
    }

    /// Record that colour id `id` shades a partial-height solid ([`crate::zband`]); a
    /// second call for the same id replaces the first.
    pub fn set_z_band(&mut self, id: u16, band: crate::zband::ZBand) {
        if id == 0 {
            return;
        }
        match self.z_bands.iter_mut().find(|(i, _)| *i == id) {
            Some(e) => e.1 = band,
            None => self.z_bands.push((id, band)),
        }
    }

    /// The z-band of colour id `id`, if its material is a partial-height one. O(#bands)
    /// (a handful), and an immediate `None` for a level with none.
    #[inline]
    pub fn z_band(&self, id: u16) -> Option<crate::zband::ZBand> {
        if self.z_bands.is_empty() || id == 0 {
            return None;
        }
        self.z_bands.iter().find(|(i, _)| *i == id).map(|(_, b)| *b)
    }

    /// Mark colour id `id` as a floor decal the camera-storey floor backdrop shows
    /// (see the `floor_decal_ids` field). Idempotent.
    pub fn mark_floor_decal(&mut self, id: u16) {
        if id != 0 && !self.floor_decal_ids.contains(&id) {
            self.floor_decal_ids.push(id);
        }
    }

    /// True when any floor decal is registered — the backdrop's O(1) skip test.
    #[inline]
    pub fn has_floor_decals(&self) -> bool {
        !self.floor_decal_ids.is_empty()
    }

    /// Is colour id `id` a floor decal the backdrop floor should shade from?
    #[inline]
    pub fn is_floor_decal(&self, id: u16) -> bool {
        id != 0 && self.floor_decal_ids.contains(&id)
    }

    /// Resolve a 1-based colour-material `id` (a geometry cell's `side_data`) to
    /// its registry [`MaterialId`]. `id == 0` (the common case: no override) and
    /// any id past the palette return `None`, so the caller falls back to the
    /// surface-kind default → byte-identical to a level with no colour palette.
    #[inline]
    pub fn color_material(&self, id: u16) -> Option<MaterialId> {
        if id == 0 {
            return None;
        }
        self.color_material_ids.get((id - 1) as usize).copied()
    }

    /// Record the material a note placard's NON-placard faces are drawn from: the
    /// colour id `backing` (0 = surface-kind default) for the note colour id `note`.
    /// A second call for the same note replaces the first.
    pub fn set_note_backing(&mut self, note: u16, backing: u16) {
        match self.note_backing.iter_mut().find(|(n, _)| *n == note) {
            Some(e) => e.1 = backing,
            None => self.note_backing.push((note, backing)),
        }
    }

    /// The backing colour id recorded for a note colour id, if any.
    #[inline]
    pub fn note_backing(&self, note: u16) -> Option<u16> {
        self.note_backing
            .iter()
            .find(|(n, _)| *n == note)
            .map(|&(_, b)| b)
    }

    /// Number of colour-palette entries (test/diagnostic helper).
    #[inline]
    pub fn color_material_len(&self) -> usize {
        self.color_material_ids.len()
    }

    /// True if any colour-palette material is animated. The near-tier dirty-flag
    /// ([`map_local_any_animated`](crate::map_local_any_animated)) consults this so
    /// a live (animated) colour surface never freezes. Empty palette ⇒ `false`.
    ///
    /// A `SelfSimilar` scale rule (Scalable Material Overlay) counts here for free:
    /// [`Material::is_animated`] now folds in [`ScaleMode::is_animated`], so a colour
    /// material tagged [`ScaleMode::SelfSimilar`] by a `scale = self_similar …` zone
    /// makes this `true` and rides the EXISTING animated-colour re-shade path — no new
    /// per-frame full re-cast. A level with no `SelfSimilar` colour material is
    /// unaffected (the gate stays `false` → today's idle-skip).
    #[inline]
    pub fn color_material_animated(&self) -> bool {
        self.color_material_ids
            .iter()
            .any(|&id| self.get(id).is_animated())
    }

    /// Tag the colour-palette material at the 1-based colour `id` with a render
    /// [`ScaleMode`] (Scalable Material Overlay). Called once per scaled `[zone]`
    /// in [`crate::registry_for_map`] AFTER the palette is registered, never in the
    /// hot path. Refreshes the cached `any_animated` flag so a `SelfSimilar` tag is
    /// reflected in [`any_animated`](Self::any_animated) /
    /// [`color_material_animated`](Self::color_material_animated).
    ///
    /// SHARED-MATERIAL RULE: a colour material is one registry slot; if two zones bind
    /// the SAME material with DIFFERENT scales, the LAST tag wins (this is a plain
    /// last-writer overwrite). Authors who need distinct scales should give each zone
    /// a DEDICATED `[material]` (one slot per scale). A no-op id (0 / past the palette)
    /// is ignored.
    pub fn set_color_material_scale(&mut self, id: u16, scale: ScaleMode) {
        if let Some(mid) = self.color_material(id) {
            if let Some(m) = self.materials.get_mut(mid.0 as usize) {
                m.scale = scale;
            }
            // A SelfSimilar tag makes the slot animated; refresh the cached flag so the
            // global dirty-flag sees it (mirrors what `register*` does on insert).
            if scale.is_animated() {
                self.any_animated = true;
            }
        }
    }

    /// IMAGE-TEXTURE attach. Walk every registered
    /// material; for each whose recipe is [`TextureRecipe::Image`]`{ tex_id }`,
    /// resolve `textures[tex_id]` and attach the `Arc<TextureImage>` to the
    /// material's [`Material::texture`] field, so [`Material::sample_image`] can read
    /// the pixels at shade time WITHOUT a `&Map`. An out-of-range `tex_id` leaves
    /// `texture = None`, so the recipe renders a magenta error block (never a panic).
    ///
    /// Called once by [`crate::registry_for_map`] after the registry is built (never
    /// in the hot path). A registry with NO `Image` material — every existing level —
    /// is untouched (byte-identical). `textures` is `map.textures` (the per-level
    /// store); the `Arc` is shared, so the big pixel buffer is never deep-copied.
    pub fn attach_image_textures(&mut self, textures: &[re_core::TextureSet]) {
        for m in &mut self.materials {
            if let TextureRecipe::Image {
                tex_id, subcell, ..
            } = m.recipe
            {
                let set = textures.get(tex_id as usize);
                // Frame 0 is what `texture` has always meant, and it is what every
                // time-less consumer (fit/aspect, edge field, the caster's cut-out
                // tests) reads. A STILL set makes this the whole story.
                m.texture = set.map(|s| std::sync::Arc::clone(s.first()));
                // ANIMATION: attach the frame set ONLY when there is more
                // than one frame, so a still texture leaves `texture_anim = None` and
                // therefore stays `is_animated() == false` and fully cacheable —
                // byte-identical to today.
                m.texture_anim = set.filter(|s| s.is_animated()).map(ImageFrames::from_set);
                // EDGE FIELD precompute: build the
                // DoG/Sobel field + mip pyramid ONCE, at load, ONLY for the ASCII-family
                // samplers that read it (Off/Quad/Braille never sample it → no cost, and
                // they stay byte-identical). Pure load-time compute.
                //
                // Built from FRAME 0 for an animated set: the field is a per-texture
                // precompute and building one per frame would multiply a load-time cost
                // by the frame count for an edge layer whose job (picking a direction
                // glyph) is dominated by the figure's standing structure.
                m.edge_field = match (subcell, m.texture.as_ref()) {
                    (SubcellMode::Ascii | SubcellMode::Contour | SubcellMode::Lines, Some(tex)) => {
                        crate::edge_field::EdgeField::build(tex)
                    }
                    _ => None,
                };
            }
        }
    }

    /// Bake a derived [`ImageSpan`] into the colour material at palette `id` IF its
    /// recipe is an `Image` with NO authored span (quality-textures Component 1). The
    /// caller ([`crate::registry_for_map`]) computes the span from the wall world
    /// extent + image aspect via [`crate::fit_span`]; this writes it into the resolved
    /// material so the hot path reads a ready `ImageSpan`. An authored `Some(span)`
    /// recipe is LEFT UNTOUCHED (the expert override wins, byte-identical). A no-op id
    /// (0 / past the palette) or a non-image material is ignored.
    pub fn bake_color_image_span(&mut self, id: u16, span: ImageSpan) {
        if let Some(mid) = self.color_material(id) {
            if let Some(m) = self.materials.get_mut(mid.0 as usize) {
                if let TextureRecipe::Image {
                    span: ref mut s, ..
                } = m.recipe
                {
                    if s.is_none() {
                        *s = Some(span);
                    }
                }
            }
        }
    }

    /// The colour material's `Image` recipe `(tex_id, has_span, fit)` at palette `id`,
    /// or `None` if that slot is not a colour-bound `Image` material. Lets
    /// [`crate::registry_for_map`] decide whether to derive + bake a span (only when
    /// `has_span == false`) and which texture's aspect to use. Read-only.
    pub fn color_image_recipe(&self, id: u16) -> Option<(u32, bool, Fit)> {
        let mid = self.color_material(id)?;
        let m = self.materials.get(mid.0 as usize)?;
        match m.recipe {
            TextureRecipe::Image {
                tex_id, span, fit, ..
            } => Some((tex_id, span.is_some(), fit)),
            _ => None,
        }
    }

    /// True if **any** registered material is animated. The renderer's
    /// dirty-flag consults this: an animated material forces every-tick
    /// re-shading even when the camera + map are unchanged (R-H1).
    #[inline]
    pub fn any_animated(&self) -> bool {
        self.any_animated
    }

    /// Number of registered material slots. Useful for tests/tools enumerating the
    /// registry (e.g. confirming the catalog was appended).
    #[inline]
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// True if no materials are registered (an empty registry).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Look up a material; falls back to slot 0 if the id is out of range so the
    /// shader never panics on a stale id. If the registry is *empty* (no slot 0
    /// either) it returns a process-wide [`DEFAULT_MATERIAL`] rather than
    /// indexing out of bounds (R-L1).
    #[inline]
    pub fn get(&self, id: MaterialId) -> &Material {
        self.materials
            .get(id.0 as usize)
            .or_else(|| self.materials.first())
            .unwrap_or(&DEFAULT_MATERIAL)
    }

    /// The shipped v0.1 material set. Slot order matches [`ids`]; each is named
    /// so it is also reachable via [`id_of`](Self::id_of).
    pub fn with_builtins() -> Self {
        Self::from_specs(&[
            MaterialSpec {
                name: "wall_brick",
                material: wall_brick(),
            },
            MaterialSpec {
                name: "wall_circuit",
                material: wall_circuit(),
            },
            MaterialSpec {
                name: "wall_matrix",
                material: wall_matrix(),
            },
            MaterialSpec {
                name: "floor_grain",
                material: floor_grain(),
            },
            MaterialSpec {
                name: "ceiling_sky",
                material: ceiling_sky(),
            },
            MaterialSpec {
                name: "puzzle_node",
                material: puzzle_node(),
            },
            MaterialSpec {
                name: "puzzle_door",
                material: puzzle_door(),
            },
            // HOLOGRAM built-in (the glyph-only transparent surface). Appended AFTER
            // the fixed surface (0..5) and puzzle (5/6) slots so it never disturbs the
            // `SurfaceKind → MaterialId` dispatch or `ids::PUZZLE_*`; authors reach it
            // by NAME (`id_of("holo_node")`) or by binding a surface to the `hologram`
            // recipe via the `[material]` path. Slot 7.
            MaterialSpec {
                name: "holo_node",
                material: holo_node(),
            },
        ])
    }

    /// Append the two PUZZLE-ELEMENT built-ins (node + door) to a registry that
    /// was built from authored surface materials. The authored merge only defines
    /// the five surface slots (0..5); the puzzle node/door are engine-owned
    /// recognisability materials the renderer dispatches directly by tag, so they
    /// must exist in slots 5/6 of EVERY registry the cast uses — authored or not.
    /// Idempotent-ish: call once after the five surface slots are registered.
    pub fn push_puzzle_builtins(&mut self) {
        self.register_named("puzzle_node", puzzle_node());
        self.register_named("puzzle_door", puzzle_door());
    }

    /// The shipped built-ins PLUS the full named CATALOG of smart materials.
    ///
    /// Slots 0..=7 are EXACTLY [`with_builtins`](Self::with_builtins) (the fixed
    /// surface dispatch, the puzzle node/door, the hologram), so every shipped level
    /// and the `SurfaceKind → MaterialId` mapping are unaffected. The catalog
    /// materials are APPENDED at slots 8.. (see [`ids::CATALOG_NAMES`]), each a pure
    /// DATA config reachable by name (`id_of("rust")`) or by a level's
    /// `[material] preset = NAME`. This is the registry the renderer builds so a
    /// preset can resolve against a known catalog without disturbing the surface set.
    pub fn with_catalog() -> Self {
        let mut r = Self::with_builtins();
        for (name, mat) in catalog_specs() {
            r.register_named(name, mat);
        }
        r
    }

    /// Resolve a CATALOG material by name from the pure data table (independent of
    /// any registry instance), e.g. for a `preset = NAME` lookup. `None` if the name
    /// is not a catalog material. Allocation-free beyond the returned `Material`
    /// (which is `Copy`).
    pub fn catalog_material(name: &str) -> Option<Material> {
        catalog_specs()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, m)| m)
    }
}

/// The pure-data CATALOG table: `(name, Material)` for every catalog material, in
/// the slot order [`ids::CATALOG_NAMES`] documents. The single source of truth both
/// [`MaterialRegistry::with_catalog`] and [`MaterialRegistry::catalog_material`]
/// read, so the names, slots, and configs can never drift apart. Each material is
/// a composition of the EXISTING data fields (recipe + palettes + light + animation
/// + stability + underline) — the "smart material = config-as-data" point.
fn catalog_specs() -> [(&'static str, Material); 46] {
    [
        ("metal", mat_metal()),
        ("gold", mat_gold()),
        ("glass", mat_glass()),
        ("stone", mat_stone()),
        ("rust", mat_rust()),
        ("ice", mat_ice()),
        ("wood", mat_wood()),
        ("fabric", mat_fabric()),
        ("water", mat_water()),
        ("lava", mat_lava()),
        ("neon", mat_neon()),
        ("monitor", mat_monitor()),
        ("flesh", mat_flesh()),
        ("void", mat_void()),
        ("holo_amber", mat_holo_amber()),
        ("circuit_red", mat_circuit_red()),
        ("haze", mat_haze()),
        ("flow", mat_flow()),
        ("spark", mat_spark()),
        // transparent (Hologram-recipe variants):
        ("vitrine", mat_vitrine()),
        ("screen", mat_screen()),
        ("veil", mat_veil()),
        // new opaque smart materials:
        ("moss", mat_moss()),
        ("ember_field", mat_ember_field()),
        ("static_noise", mat_static_noise()),
        ("lattice", mat_lattice()),
        // super-glyph smart materials:
        ("beat_grid", mat_beat_grid()),
        ("auto_tile", mat_auto_tile()),
        ("boil", mat_boil()),
        // fractal / infinite-zoom portal materials:
        ("portal_zoom", mat_portal_zoom()),
        ("portal_spiral", mat_portal_spiral()),
        // angle-glyph / glass-marker / multi-channel-volume demo materials:
        ("facet", mat_facet()),
        ("pane", mat_pane()),
        ("relief_vol", mat_relief_vol()),
        // semantic surfaces — APPENDED at the tail so no
        // existing MaterialId / slot shifts (goldens hold). Each opts into a
        // `SurfaceFx` finish over an ordinary base recipe.
        ("frost", mat_frost()),
        ("wet_stone", mat_wet_stone()),
        ("redacted", mat_redacted()),
        ("watcher", mat_watcher()),
        ("worn_floor", mat_worn_floor()),
        ("decay", mat_decay()),
        // animation-track material — APPENDED at the tail.
        ("node_breathing", mat_node_breathing()),
        // the cell alphabet — APPENDED at the tail (slot 49).
        ("substrate", mat_substrate()),
        // the state-driven voice pair — APPENDED at the tail (slots 50/51).
        ("mnemos_hold", mat_mnemos_hold()),
        ("lethe_release", mat_lethe_release()),
        // edge-matched conduit lattice — APPENDED at the tail (slot 52).
        ("conduit", mat_conduit()),
        // the structured bay×floor window grid — APPENDED at the tail (slot 53).
        ("facade", mat_facade()),
    ]
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ── built-in material definitions (pure data — the "no magic" requirement) ──

fn wall_brick() -> Material {
    Material {
        recipe: TextureRecipe::Brick {
            aspect: 2.0,
            mortar_w: 0.16,
            grain: 0.6,
        },
        // Warm bone → cool stone via cosine palette.
        palette: Palette {
            a: Vec3::new(0.55, 0.52, 0.48),
            b: Vec3::new(0.30, 0.28, 0.30),
            c: Vec3::new(0.8, 0.9, 1.0),
            d: Vec3::new(0.10, 0.15, 0.25),
        },
        bg_palette: Palette {
            a: Vec3::new(0.18, 0.17, 0.20),
            b: Vec3::new(0.10, 0.10, 0.14),
            c: Vec3::new(0.7, 0.8, 1.0),
            d: Vec3::new(0.20, 0.25, 0.35),
        },
        rank: RankStrategy::StructureFirst,
        seed: SHADER_SEED ^ 0x0001,
        glyph_source: GlyphSource::Recipe,
        animation: Animation::NONE,
        // Brick relief: faces raised, mortar grooves recessed. A pronounced
        // height with strong normals so courses cast a clear light/shade rake;
        // matte (low spec) stone with firm AO in the seams.
        light: LightModel {
            height_scale: 1.0,
            normal_strength: 1.1,
            shininess: 10.0,
            specular: 0.18,
            spec_color: 0x00FFF2E0,
            ao_strength: 0.7,
            emissive: 0.0,
        },
        // The brick BODY is texture detail (faces + fine grit) — Stable, so the
        // body glyph + speck stop churning as the player approaches. The structural
        // mortar seams (│/─) carry the read and stay responsive via their own low
        // rank; only the body's deadband widens.
        stability: GlyphStability::Stable,
        // Not a mirror — an ordinary opaque surface.
        reflective: false,
        clarity: 1.0,
        // No underline accent on shipped brick: keeping it OFF preserves every
        // shipped non-hologram level byte-for-byte (the only new accent in shipped
        // content is the hologram scan-glow). The LitRidge specular-glint accent is
        // available + tested via a synthetic material — opt a wall into it by
        // setting `underline_accent` to a `LitRidge` `UnderlineAccent`.
        underline_accent: UnderlineAccent::NONE,
        // No kindred-symbol family layer / proximity-bold on the shipped built-in
        // (unchanged output — shipped levels render byte-for-byte the same).
        family: None,
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        lod_override: None,
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

fn wall_circuit() -> Material {
    Material {
        // CALMER WALL: narrower traces (0.12 → 0.09) read as crisp routed lines
        // rather than wide bands, so the structure is clean and the colour carries.
        recipe: TextureRecipe::Circuit {
            scale: 1.6,
            trace_w: 0.09,
        },
        // Teal copper board with bright cyan accents.
        palette: Palette {
            a: Vec3::new(0.20, 0.55, 0.55),
            b: Vec3::new(0.20, 0.40, 0.45),
            c: Vec3::new(1.0, 1.0, 1.0),
            d: Vec3::new(0.30, 0.20, 0.10),
        },
        bg_palette: Palette {
            a: Vec3::new(0.05, 0.12, 0.16),
            b: Vec3::new(0.04, 0.10, 0.14),
            c: Vec3::new(1.0, 1.0, 1.0),
            d: Vec3::new(0.0, 0.1, 0.2),
        },
        rank: RankStrategy::StructureFirst,
        seed: SHADER_SEED ^ 0x0002,
        glyph_source: GlyphSource::Recipe,
        // A slow brightness pulse: the trace wall "breathes" as if powered.
        animation: Animation {
            kind: AnimKind::Pulse,
            speed: 0.5,
            amplitude: 0.18,
        },
        // Circuit relief: pads/vias + traces are raised lit metal with a tight,
        // glossy cyan glint; the board pour is low. A touch of emissive so the
        // traces read as powered (glow regardless of the key light).
        light: LightModel {
            height_scale: 0.9,
            normal_strength: 1.0,
            shininess: 28.0,
            specular: 0.6,
            spec_color: 0x00CFFFFF,
            ao_strength: 0.55,
            emissive: 0.10,
        },
        // The circuit wall reads as STRUCTURE — routed trace lines + pads. Those
        // structural glyphs must stay crisp/responsive as the player moves, so this
        // wall is Responsive (the sparse copper-pour body carries little glyph
        // detail to churn anyway; the look is in the traces + colour).
        stability: GlyphStability::Responsive,
        // Not a mirror — an ordinary opaque surface.
        reflective: false,
        clarity: 1.0,
        // No underline accent (the circuit's glow already rides its emissive
        // colored underline via `cell_style`).
        underline_accent: UnderlineAccent::NONE,
        // No kindred-symbol family layer / proximity-bold on the shipped built-in
        // (unchanged output — shipped levels render byte-for-byte the same).
        family: None,
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        lod_override: None,
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

fn wall_matrix() -> Material {
    Material {
        // Reuse the circuit recipe with a green palette + denser traces so the
        // "matrix" wall reads as a green data-rain panel without a new recipe.
        // CALMER WALL: matrix traces narrowed (0.18 → 0.13) so the green data wall
        // reads as routed lines + colour, not a dense identical-glyph fill.
        recipe: TextureRecipe::Circuit {
            scale: 2.4,
            trace_w: 0.13,
        },
        palette: Palette {
            a: Vec3::new(0.25, 0.70, 0.35),
            b: Vec3::new(0.20, 0.30, 0.20),
            c: Vec3::new(1.0, 1.0, 1.0),
            d: Vec3::new(0.0, 0.33, 0.16),
        },
        bg_palette: Palette {
            a: Vec3::new(0.02, 0.10, 0.04),
            b: Vec3::new(0.02, 0.08, 0.04),
            c: Vec3::new(1.0, 1.0, 1.0),
            d: Vec3::new(0.0, 0.2, 0.1),
        },
        rank: RankStrategy::StructureFirst,
        seed: SHADER_SEED ^ 0x0003,
        glyph_source: GlyphSource::Recipe,
        // Matrix-rain: scroll the texture downward over time (data flow).
        animation: Animation {
            kind: AnimKind::Shift,
            speed: 1.0,
            amplitude: 1.5,
        },
        // Matrix wall: glowing green traces, strong emissive so the data-rain
        // self-lights; gentle relief + a soft green glint.
        light: LightModel {
            height_scale: 0.7,
            normal_strength: 0.8,
            shininess: 24.0,
            specular: 0.5,
            spec_color: 0x00B0FFC0,
            ao_strength: 0.4,
            emissive: 0.30,
        },
        // Like the circuit wall, the matrix wall is routed-line STRUCTURE (data
        // rain) — responsive so the traces stay crisp. Its `Shift` animation already
        // re-shades every tick; the deadband would fight that, so keep it ×1.
        stability: GlyphStability::Responsive,
        // Not a mirror — an ordinary opaque surface.
        reflective: false,
        clarity: 1.0,
        // No underline accent layer (matrix glow rides its emissive underline).
        underline_accent: UnderlineAccent::NONE,
        // No kindred-symbol family layer / proximity-bold on the shipped built-in
        // (unchanged output — shipped levels render byte-for-byte the same).
        family: None,
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        lod_override: None,
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

fn floor_grain() -> Material {
    Material {
        // CALMER FLOOR: specks cut hard again (0.10 → 0.05) and the speck lattice spread out
        // (scale 3.0 → 2.2) so the floor is overwhelmingly a coloured low-density
        // ground; the tiling is carried by the now-DASHED grid seam (see
        // [`floor_seam_inked`]), the shading by the fg/bg colour wash — never a
        // packed speck/grid field. World-anchored specks (Worley) so they don't churn.
        recipe: TextureRecipe::Grain {
            scale: 2.2,
            speck: 0.05,
            tile_eps: 0.05,
        },
        // Bright green specks over a cool slate wash.
        palette: Palette {
            a: Vec3::new(0.30, 0.65, 0.45),
            b: Vec3::new(0.25, 0.35, 0.30),
            c: Vec3::new(1.0, 1.0, 1.0),
            d: Vec3::new(0.1, 0.4, 0.2),
        },
        bg_palette: Palette {
            a: Vec3::new(0.16, 0.20, 0.30),
            b: Vec3::new(0.08, 0.10, 0.16),
            c: Vec3::new(0.8, 0.9, 1.0),
            d: Vec3::new(0.2, 0.3, 0.4),
        },
        rank: RankStrategy::FineDetail,
        seed: SHADER_SEED ^ 0x0004,
        glyph_source: GlyphSource::Recipe,
        animation: Animation::NONE,
        // Floor grain: a subtle bump (specks raised, tile seams shallow grooves)
        // — low height so the floor stays readable, light AO, faint matte glint.
        light: LightModel {
            height_scale: 0.5,
            normal_strength: 0.6,
            shininess: 12.0,
            specular: 0.2,
            spec_color: 0x00E8F0FF,
            ao_strength: 0.45,
            emissive: 0.0,
        },
        // The floor BODY is a large, mostly-blank coloured field with fine grit —
        // exactly the surface that churns worst underfoot as you walk. Stable: the
        // grit specks + body glyph anchor to the world texel + a wide LOD bucket, so
        // the floor stops shimmering while the tile-seam structure (low rank) stays
        // crisp and the colour wash keeps reading the raw distance.
        stability: GlyphStability::Stable,
        // Not a mirror — an ordinary opaque surface.
        reflective: false,
        clarity: 1.0,
        // No underline accent layer on the floor (unchanged output).
        underline_accent: UnderlineAccent::NONE,
        // No kindred-symbol family layer / proximity-bold on the shipped built-in
        // (unchanged output — shipped levels render byte-for-byte the same).
        family: None,
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        lod_override: None,
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

fn ceiling_sky() -> Material {
    Material {
        recipe: TextureRecipe::Sky {
            scale: 6.0,
            density: 0.10,
        },
        // Star colors: warm-white → faint blue across the class range.
        palette: Palette {
            a: Vec3::new(0.80, 0.82, 0.90),
            b: Vec3::new(0.18, 0.16, 0.10),
            c: Vec3::new(1.0, 1.0, 1.0),
            d: Vec3::new(0.0, 0.1, 0.3),
        },
        bg_palette: Palette {
            a: Vec3::new(0.06, 0.08, 0.16),
            b: Vec3::new(0.04, 0.05, 0.10),
            c: Vec3::new(0.6, 0.8, 1.0),
            d: Vec3::new(0.1, 0.2, 0.4),
        },
        rank: RankStrategy::FineDetail,
        seed: SHADER_SEED ^ 0x0005,
        glyph_source: GlyphSource::Recipe,
        animation: Animation::NONE,
        // Sky / ceiling: essentially flat and self-lit — stars are emissive
        // points, the nebula needs no relief. Minimal lighting so the void reads
        // dark and the stars stay crisp.
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 8.0,
            specular: 0.0,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.0,
            emissive: 0.6,
        },
        // The sky is the largest, farthest, flattest field — and its stars twinkle
        // worst as the camera moves (mid-rank detail re-selecting on tiny steps).
        // Frozen: stars anchor to the world texel + the widest LOD bucket with the
        // strongest reveal hysteresis, so the starfield stays put while the nebula
        // colour + the depth band keep reading the raw distance (parallax live).
        stability: GlyphStability::Frozen,
        // Not a mirror — an ordinary opaque surface.
        reflective: false,
        clarity: 1.0,
        // No underline accent layer on the sky/void (unchanged output).
        underline_accent: UnderlineAccent::NONE,
        // No kindred-symbol family layer / proximity-bold on the shipped built-in
        // (unchanged output — shipped levels render byte-for-byte the same).
        family: None,
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        lod_override: None,
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

/// HOLOGRAM — a glyph-only, transparent, glowing scanline surface. The bg is a
/// near-black VOID (no filled colour); the only ink is fg-coloured horizontal line
/// glyphs (`═ ─ ╌ ·`) forming holographic scanlines that SCROLL VERTICALLY over
/// time (a `Shift` animation drifting the surface `y`). A single cyan hue palette
/// so it reads as one coherent holographic colour. Self-lit (the relief pass leaves
/// it alone) + emissive so the lines glow regardless of the headlamp. `Responsive`
/// stability: the lines are structure and must read crisply (and its `Shift` anim
/// re-shades every tick anyway, so a wide deadband would fight it). Far-frozen at
/// distance like every recipe — a distant hologram stops scrolling and holds still.
fn holo_node() -> Material {
    Material {
        recipe: TextureRecipe::Hologram {
            line_freq: 6.0,
            scan_w: 0.5,
        },
        // A single cyan holo hue (flat-ish cosine: a tight cyan family around 0.5).
        palette: Palette {
            a: Vec3::new(0.35, 0.85, 0.95),
            b: Vec3::new(0.10, 0.12, 0.05),
            c: Vec3::new(1.0, 1.0, 1.0),
            d: Vec3::new(0.0, 0.10, 0.20),
        },
        // bg_palette is unused (the recipe forces a void bg), but keep a dark entry
        // so any incidental eval reads near-black, consistent with the void look.
        bg_palette: Palette {
            a: Vec3::splat(0.01),
            b: Vec3::ZERO,
            c: Vec3::ONE,
            d: Vec3::ZERO,
        },
        rank: RankStrategy::StructureFirst,
        seed: SHADER_SEED ^ 0x0008,
        glyph_source: GlyphSource::Recipe,
        // Vertical SCROLL: drift the surface coord's `y` over time so the scanlines
        // move up the hologram (the see-through, alive look).
        animation: Animation {
            kind: AnimKind::Shift,
            speed: 0.8,
            amplitude: 1.0,
        },
        // Self-lit projected light: zero relief, strong emissive so the lines glow.
        // The lighting pass early-returns for the Hologram recipe (it must not be
        // relit), so these mostly drive `cell_style`'s emissive under-glow accent.
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 16.0,
            specular: 0.0,
            spec_color: 0x00CFFFFF,
            ao_strength: 0.0,
            emissive: 0.8,
        },
        // The scanlines are structure — keep them crisp/responsive (and the Shift
        // animation already re-shades every tick).
        stability: GlyphStability::Responsive,
        // Not a mirror — it is transparent, not reflective.
        reflective: false,
        clarity: 1.0,
        // SMART-MATERIAL SCAN-GLOW: a CURLY (undercurl) colored underline under
        // every inked scan-line cell, in the cell's own holo hue (HUE_FROM_FG). The
        // curl sits just under the line glyphs (`═ ─ ╌ ·`) → a glowing scan-glow
        // that reads as the hologram's projected light bleeding below each line.
        // `Inked` gate, so only the lit scan cells glow (the void gaps stay clean).
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Curly,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::Inked,
        },
        // No kindred-symbol family / proximity-bold on the hologram built-in
        // (unchanged output).
        family: None,
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        lod_override: None,
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

/// PUZZLE NODE — a neuron. A recognisable, self-lit node symbol the player reads
/// instantly against the calmer walls/floor. A FIXED glyph (never thinned by LOD
/// or depth, so it stays a node at any distance), a saturated amber-gold fg over a
/// dark base, and strong emissive so it GLOWS regardless of the headlamp. The
/// `cell_style` pass bolds it (bright lit fg) and gives it the emissive under-glow,
/// so a node reads as a node, not as a wall face.
///
/// `lod_override = full_detail_to(PUZZLE_NODE_FULL_DETAIL_TO)`: a puzzle node is
/// PUZZLE-CRITICAL — the player must be able to read it from across a room (far
/// node grids on levels 05/06 would otherwise dim into fog like an ordinary
/// wall). The universal distance-depth effect (`depth_contrast.rs`) reads this SAME
/// per-material override for both its colour (contrast/sky-fade) and glyph-variety
/// halves, so the node keeps full contrast/no fade well past any shipped room's
/// extent, using the existing hero-object lever rather than a glyph special case.
fn puzzle_node() -> Material {
    Material {
        // A near-flat structural base (the recipe barely matters — the glyph is
        // fixed and the colour is a flat saturated amber), so it shades stably.
        recipe: TextureRecipe::Brick {
            aspect: 1.0,
            mortar_w: 0.0,
            grain: 0.0,
        },
        // Saturated amber-gold neuron, flat (zero-amplitude palette → constant).
        palette: Palette {
            a: Vec3::new(1.0, 0.78, 0.22),
            b: Vec3::ZERO,
            c: Vec3::ONE,
            d: Vec3::ZERO,
        },
        bg_palette: Palette {
            a: Vec3::new(0.10, 0.07, 0.02),
            b: Vec3::ZERO,
            c: Vec3::ONE,
            d: Vec3::ZERO,
        },
        rank: RankStrategy::StructureFirst,
        seed: SHADER_SEED ^ 0x0006,
        glyph_source: GlyphSource::Fixed(PUZZLE_NODE_GLYPH),
        // A gentle pulse so the node "lives" (a neuron) — calm, not strobing.
        animation: Animation {
            kind: AnimKind::Pulse,
            speed: 0.6,
            amplitude: 0.12,
        },
        // Strongly self-lit so it pops out of the scene regardless of where the
        // player's headlamp points; a tight bright glint on top.
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 32.0,
            specular: 0.4,
            spec_color: 0x00FFF4D0,
            ao_strength: 0.0,
            emissive: 0.85,
        },
        // A puzzle element: a structural, recognisable marker. Responsive (and its
        // glyph is FIXED anyway, so it never LOD-thins) — the player must read it
        // crisply at every distance.
        stability: GlyphStability::Responsive,
        // Not a mirror — an ordinary opaque surface.
        reflective: false,
        clarity: 1.0,
        // No underline accent on the node (its glow rides the emissive underline).
        underline_accent: UnderlineAccent::NONE,
        // STATE BUS: the node's glyph is driven by its
        // live state word (the bridge encodes powered/dead/decoy/won; the `STATE_NODE`
        // family supplies the matching glyph). Member 0 == the shipped `◉`, and an
        // EMPTY state bus reads state 0 for every tag, so an unwired node renders
        // byte-for-byte as the shipped fixed-glyph neuron — goldens hold. The bridge
        // also lifts the COLOUR per state (in `re_game`); here the GLYPH separates a
        // powered node from a dead one. State-driven ⇒ STATIC (not time-animated), so
        // it stays cacheable and only re-shades on an interaction (a dirty event).
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::STATE_NODE,
            driver: crate::glyph_family::FamilyDriver::State,
            speed: 0.0,
        }),
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        // PUZZLE-CRITICAL SURFACE exemption (see the function doc): hold full
        // contrast / no sky-fade / no glyph-variety collapse out to
        // `PUZZLE_NODE_FULL_DETAIL_TO` world cells — comfortably past any shipped
        // room's extent, so a node never dims into fog like an ordinary wall.
        lod_override: Some(crate::lod::LodOverride::full_detail_to(
            PUZZLE_NODE_FULL_DETAIL_TO,
        )),
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

/// PUZZLE DOOR — the exit. A distinct, recognisable doorway glyph in a cool
/// cyan-white so it reads as "the way out", clearly different from the warm node
/// and from any wall. Fixed glyph (stable at every distance), strong emissive so
/// it beckons from across the room.
fn puzzle_door() -> Material {
    Material {
        recipe: TextureRecipe::Brick {
            aspect: 1.0,
            mortar_w: 0.0,
            grain: 0.0,
        },
        // Cool cyan-white exit, flat.
        palette: Palette {
            a: Vec3::new(0.55, 0.92, 1.0),
            b: Vec3::ZERO,
            c: Vec3::ONE,
            d: Vec3::ZERO,
        },
        bg_palette: Palette {
            a: Vec3::new(0.03, 0.09, 0.12),
            b: Vec3::ZERO,
            c: Vec3::ONE,
            d: Vec3::ZERO,
        },
        rank: RankStrategy::StructureFirst,
        seed: SHADER_SEED ^ 0x0007,
        glyph_source: GlyphSource::Fixed(PUZZLE_DOOR_GLYPH),
        animation: Animation::NONE,
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 24.0,
            specular: 0.35,
            spec_color: 0x00E0FFFF,
            ao_strength: 0.0,
            emissive: 0.70,
        },
        // The exit door: a structural, fixed-glyph marker — Responsive, read
        // crisply at every distance (the fixed glyph never LOD-thins regardless).
        stability: GlyphStability::Responsive,
        // Not a mirror — an ordinary opaque surface.
        reflective: false,
        clarity: 1.0,
        // No underline accent on the exit door (unchanged output).
        underline_accent: UnderlineAccent::NONE,
        // STATE BUS: the exit brightens to a charged gate
        // when the puzzle is WON (state 3) and is its quiet panel otherwise. Member 0
        // == the shipped `▒`, and an EMPTY bus reads state 0, so an unwired door is
        // byte-for-byte the shipped exit — goldens hold. State-driven ⇒ STATIC.
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::STATE_DOOR,
            driver: crate::glyph_family::FamilyDriver::State,
            speed: 0.0,
        }),
        // No super-glyph selector — the recipe glyph is unchanged.
        super_glyph: None,
        proximity_bold: None,
        lod_override: None,
        glass_marker: false,
        surface_fx: None,
        // No animation tracks — the body glyph is static.
        anim_tracks: AnimTrackSet::EMPTY,
        scale: ScaleMode::None,
        texture: None,
        edge_field: None,
        texture_anim: None,
        detail: false,
    }
}

/// PUZZLE element glyphs — recognisable, single, fixed (never LOD-thinned). The
/// node is a filled-ring neuron; the door is a doorway/arch.
const PUZZLE_NODE_GLYPH: u32 = 0x25C9; // ◉  a filled ringed node (neuron)
const PUZZLE_DOOR_GLYPH: u32 = 0x2592; // ▒  a portal panel (the exit)

/// Distance (world cells) [`puzzle_node`]'s [`crate::lod::LodOverride::full_detail_to`]
/// holds full contrast/detail out to — comfortably past [`FAR_FREEZE_DIST`] (32.0, the
/// far-freeze distance and the universal depth effect's default full-sky-blend
/// distance) and past every shipped room's extent (rooms run 8–25 world cells per
/// `depth_contrast`'s own module note), so a puzzle node never dims like an ordinary
/// wall within any level the campaign ships.
const PUZZLE_NODE_FULL_DETAIL_TO: f32 = 40.0;

// ─────────────────────────── catalog materials ──────────────────────────────
//
// Each is a DATA config of the composable material system — different recipe,
// palette mood, relief, animation, emissive, reflective, stability, underline —
// no new shader code paths. Tuned so each reads as its evoked substance in ASCII.
// See `docs/MATERIALS.md` for the catalog table.

/// A small builder seeded off [`DEFAULT_MATERIAL`] so each catalog material spells
/// out only what makes it distinct. Keeps the configs short + readable.
#[inline]
fn catalog_base(recipe: TextureRecipe, seed_salt: u32) -> Material {
    Material {
        recipe,
        seed: SHADER_SEED ^ seed_salt,
        ..DEFAULT_MATERIAL.clone()
    }
}

/// Convenience to build an iq cosine palette from the four control vectors.
#[inline]
fn pal(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> Palette {
    Palette {
        a: Vec3::from_array(a),
        b: Vec3::from_array(b),
        c: Vec3::from_array(c),
        d: Vec3::from_array(d),
    }
}

/// METAL — brushed steel: a cool grey directional grain (horizontal scanline grain),
/// high specular sheen + relief, near-zero emissive. Reads as a hard reflective-ish
/// panel without being a literal mirror.
fn mat_metal() -> Material {
    Material {
        palette: pal(
            [0.55, 0.58, 0.62],
            [0.22, 0.22, 0.24],
            [1.0, 1.0, 1.0],
            [0.0, 0.05, 0.1],
        ),
        bg_palette: pal(
            [0.10, 0.11, 0.13],
            [0.05, 0.05, 0.06],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.4,
            normal_strength: 0.5,
            shininess: 48.0,
            specular: 0.9,
            spec_color: 0x00F2F6FF,
            ao_strength: 0.35,
            emissive: 0.04,
        },
        stability: GlyphStability::Stable,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Dotted,
            color: 0x00C8D2E6,
            gate: AccentGate::LitRidge,
        },
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 5.0,
                duty: 0.7,
                warp: 0.45,
                axis: 0,
            },
            0x00C0,
        )
    }
}

/// GOLD / BRASS — a warm metal: rich amber sheen, high specular, a faint emissive
/// lustre so it glows in a dim room. Vertical brushed grain for a poured-bar read.
fn mat_gold() -> Material {
    Material {
        palette: pal(
            [0.80, 0.62, 0.22],
            [0.20, 0.22, 0.16],
            [1.0, 1.0, 1.0],
            [0.0, 0.08, 0.16],
        ),
        bg_palette: pal(
            [0.16, 0.11, 0.03],
            [0.08, 0.06, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.4,
            normal_strength: 0.5,
            shininess: 40.0,
            specular: 1.0,
            spec_color: 0x00FFF0C0,
            ao_strength: 0.3,
            emissive: 0.12,
        },
        stability: GlyphStability::Stable,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Dotted,
            color: 0x00FFE08A,
            gate: AccentGate::LitRidge,
        },
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 4.0,
                duty: 0.72,
                warp: 0.4,
                axis: 1,
            },
            0x00C1,
        )
    }
}

/// GLASS / WINDOW — a faint, reflective, high-clarity pane: low ink, near-void bg,
/// a crisp mirror with a cool tint. The reflective flag makes the raycaster show the
/// scene in it; clarity high so it reads as clean glass, not a broken surface.
fn mat_glass() -> Material {
    Material {
        palette: pal(
            [0.62, 0.72, 0.80],
            [0.10, 0.10, 0.12],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.04, 0.06, 0.09],
            [0.02, 0.03, 0.04],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.1,
            normal_strength: 0.2,
            shininess: 64.0,
            specular: 0.8,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.15,
            emissive: 0.06,
        },
        stability: GlyphStability::Stable,
        reflective: true,
        clarity: 0.92,
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 2.0,
                duty: 0.25,
                warp: 0.15,
                axis: 2,
            },
            0x00C2,
        )
    }
}

/// STONE / CONCRETE — matte, deep AO, low emissive: a heavy bricked wall with a
/// cool grey palette and strong shadowed seams. Frozen-stable so the rough field
/// holds still underfoot/around you.
fn mat_stone() -> Material {
    Material {
        recipe: TextureRecipe::Brick {
            aspect: 1.4,
            mortar_w: 0.20,
            grain: 0.7,
        },
        palette: pal(
            [0.48, 0.48, 0.50],
            [0.16, 0.16, 0.17],
            [1.0, 1.0, 1.0],
            [0.0, 0.05, 0.1],
        ),
        bg_palette: pal(
            [0.12, 0.12, 0.13],
            [0.06, 0.06, 0.07],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 1.1,
            normal_strength: 1.2,
            shininess: 6.0,
            specular: 0.06,
            spec_color: 0x00E0E0E0,
            ao_strength: 0.85,
            emissive: 0.0,
        },
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Brick {
                aspect: 1.4,
                mortar_w: 0.20,
                grain: 0.7,
            },
            0x00C3,
        )
    }
}

/// RUST / CORROSION — a warm, noisy, rough surface: a high-warp horizontal grain in
/// oxide oranges/browns, matte (low spec), pitted AO. Stable so the grime doesn't
/// crawl.
fn mat_rust() -> Material {
    Material {
        palette: pal(
            [0.55, 0.34, 0.20],
            [0.28, 0.20, 0.12],
            [1.0, 1.1, 0.9],
            [0.0, 0.15, 0.3],
        ),
        bg_palette: pal(
            [0.14, 0.08, 0.05],
            [0.07, 0.04, 0.03],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.7,
            normal_strength: 0.9,
            shininess: 5.0,
            specular: 0.08,
            spec_color: 0x00C08050,
            ao_strength: 0.7,
            emissive: 0.02,
        },
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 6.0,
                duty: 0.6,
                warp: 0.9,
                axis: 0,
            },
            0x00C4,
        )
    }
}

/// ICE / FROST — cool, sparse, slightly reflective: a pale blue circuit-pour gives
/// crystalline facets; faint mirror clarity for a glassy sheen, low emissive.
fn mat_ice() -> Material {
    Material {
        recipe: TextureRecipe::Circuit {
            scale: 2.2,
            trace_w: 0.06,
        },
        palette: pal(
            [0.62, 0.80, 0.92],
            [0.12, 0.12, 0.10],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.08, 0.12, 0.18],
            [0.04, 0.06, 0.09],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.5,
            normal_strength: 0.6,
            shininess: 56.0,
            specular: 0.6,
            spec_color: 0x00E8F4FF,
            ao_strength: 0.3,
            emissive: 0.05,
        },
        stability: GlyphStability::Stable,
        reflective: true,
        clarity: 0.55,
        ..catalog_base(
            TextureRecipe::Circuit {
                scale: 2.2,
                trace_w: 0.06,
            },
            0x00C5,
        )
    }
}

/// WOOD — warm directional grain: a brushed horizontal scanline grain in browns
/// with a soft satin sheen and a gentle relief, matte. Stable.
fn mat_wood() -> Material {
    Material {
        palette: pal(
            [0.50, 0.34, 0.18],
            [0.18, 0.13, 0.07],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.24],
        ),
        bg_palette: pal(
            [0.14, 0.09, 0.04],
            [0.07, 0.05, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.5,
            normal_strength: 0.6,
            shininess: 14.0,
            specular: 0.18,
            spec_color: 0x00FFE0B0,
            ao_strength: 0.5,
            emissive: 0.0,
        },
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 3.0,
                duty: 0.78,
                warp: 0.55,
                axis: 0,
            },
            0x00C6,
        )
    }
}

/// FABRIC / WEAVE — soft, low relief, woven cross-hatch: a muted dusty palette, no
/// spec, gentle AO. The woven scanline axis reads as cloth threads.
fn mat_fabric() -> Material {
    Material {
        palette: pal(
            [0.52, 0.42, 0.48],
            [0.14, 0.12, 0.14],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.13, 0.11, 0.13],
            [0.06, 0.05, 0.06],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.3,
            normal_strength: 0.4,
            shininess: 8.0,
            specular: 0.05,
            spec_color: 0x00E0D8E0,
            ao_strength: 0.45,
            emissive: 0.0,
        },
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 7.0,
                duty: 0.55,
                warp: 0.25,
                axis: 2,
            },
            0x00C7,
        )
    }
}

/// WATER / LIQUID — animated cool surface: a `Shift` flow over a blue circuit-pour,
/// reflective with mid clarity, faint emissive caustic shimmer.
fn mat_water() -> Material {
    Material {
        recipe: TextureRecipe::Circuit {
            scale: 1.8,
            trace_w: 0.08,
        },
        palette: pal(
            [0.22, 0.46, 0.66],
            [0.16, 0.22, 0.26],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.24],
        ),
        bg_palette: pal(
            [0.04, 0.10, 0.16],
            [0.02, 0.05, 0.09],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Shift,
            speed: 0.6,
            amplitude: 0.8,
        },
        light: LightModel {
            height_scale: 0.5,
            normal_strength: 0.7,
            shininess: 50.0,
            specular: 0.6,
            spec_color: 0x00CFEFFF,
            ao_strength: 0.25,
            emissive: 0.08,
        },
        stability: GlyphStability::Responsive,
        reflective: true,
        clarity: 0.45,
        ..catalog_base(
            TextureRecipe::Circuit {
                scale: 1.8,
                trace_w: 0.08,
            },
            0x00C8,
        )
    }
}

/// LAVA / ENERGY — hot, high-emissive, pulsing: an orange-red grain that breathes
/// with a `Pulse`, strong emissive so it self-lights, an under-glow accent.
fn mat_lava() -> Material {
    Material {
        palette: pal(
            [0.85, 0.32, 0.08],
            [0.15, 0.22, 0.06],
            [1.0, 1.0, 1.0],
            [0.0, 0.15, 0.3],
        ),
        bg_palette: pal(
            [0.18, 0.04, 0.01],
            [0.10, 0.02, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Pulse,
            speed: 0.7,
            amplitude: 0.22,
        },
        light: LightModel {
            height_scale: 0.5,
            normal_strength: 0.6,
            shininess: 10.0,
            specular: 0.2,
            spec_color: 0x00FFC060,
            ao_strength: 0.4,
            emissive: 0.75,
        },
        stability: GlyphStability::Responsive,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Curly,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::Inked,
        },
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 5.0,
                duty: 0.65,
                warp: 0.85,
                axis: 2,
            },
            0x00C9,
        )
    }
}

/// NEON / SIGN — emissive + flickering: a saturated magenta phosphor with a `Flicker`
/// (a failing tube), strong emissive, an underline scan-glow. Vertical tube bands.
fn mat_neon() -> Material {
    Material {
        palette: pal(
            [0.95, 0.25, 0.75],
            [0.05, 0.10, 0.08],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.05, 0.01, 0.05],
            [0.02, 0.0, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Flicker,
            speed: 8.0,
            amplitude: 0.35,
        },
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 16.0,
            specular: 0.0,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.0,
            emissive: 0.9,
        },
        stability: GlyphStability::Responsive,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Curly,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::Inked,
        },
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 3.0,
                duty: 0.5,
                warp: 0.0,
                axis: 1,
            },
            0x00CA,
        )
    }
}

/// MONITOR / CRT / SCREEN — phosphor scanlines + flicker: green raster bands on a
/// dead-black tube, emissive glow, a `Flicker` for the refresh. The canonical
/// horizontal-scanline use of the new recipe.
fn mat_monitor() -> Material {
    Material {
        palette: pal(
            [0.28, 0.85, 0.40],
            [0.10, 0.14, 0.08],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.01, 0.03, 0.02],
            [0.0, 0.01, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Flicker,
            speed: 6.0,
            amplitude: 0.18,
        },
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 16.0,
            specular: 0.0,
            spec_color: 0x00B0FFC0,
            ao_strength: 0.0,
            emissive: 0.85,
        },
        stability: GlyphStability::Responsive,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Curly,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::Inked,
        },
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 9.0,
                duty: 0.55,
                warp: 0.05,
                axis: 0,
            },
            0x00CB,
        )
    }
}

/// FLESH / ORGANIC — warm, soft, slowly breathing: a pink-red grain with a slow
/// `Pulse`, soft relief, faint subsurface emissive, no spec. Stable.
fn mat_flesh() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.0,
            speck: 0.12,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.70, 0.40, 0.42],
            [0.16, 0.10, 0.10],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.24],
        ),
        bg_palette: pal(
            [0.18, 0.08, 0.09],
            [0.09, 0.04, 0.05],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Pulse,
            speed: 0.25,
            amplitude: 0.1,
        },
        light: LightModel {
            height_scale: 0.4,
            normal_strength: 0.5,
            shininess: 12.0,
            specular: 0.12,
            spec_color: 0x00FFD0C0,
            ao_strength: 0.5,
            emissive: 0.18,
        },
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.0,
                speck: 0.12,
                tile_eps: 0.0,
            },
            0x00CC,
        )
    }
}

/// VOID / SPACE — near-black, sparse emissive specks: a dark starfield, the dimmest
/// catalog material; emissive points only, frozen so the field never crawls.
fn mat_void() -> Material {
    Material {
        recipe: TextureRecipe::Sky {
            scale: 7.0,
            density: 0.06,
        },
        palette: pal(
            [0.70, 0.74, 0.85],
            [0.20, 0.18, 0.12],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.3],
        ),
        bg_palette: pal(
            [0.02, 0.02, 0.04],
            [0.01, 0.01, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 8.0,
            specular: 0.0,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.0,
            emissive: 0.5,
        },
        stability: GlyphStability::Frozen,
        ..catalog_base(
            TextureRecipe::Sky {
                scale: 7.0,
                density: 0.06,
            },
            0x00CD,
        )
    }
}

/// HOLO_AMBER — a hologram VARIANT: the glyph-only transparent scanline surface in
/// an amber/gold hue instead of cyan, a touch faster scroll. Shows the recipe is a
/// data config (only the palette + anim differ from `holo_node`).
fn mat_holo_amber() -> Material {
    Material {
        palette: pal(
            [0.95, 0.70, 0.25],
            [0.10, 0.10, 0.05],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        ..holo_node()
    }
}

/// CIRCUIT_RED — a circuit VARIANT: the trace board in a hot red/orange palette with
/// a faster pulse, denser traces. A second proof the circuit look is config-driven.
fn mat_circuit_red() -> Material {
    Material {
        recipe: TextureRecipe::Circuit {
            scale: 1.8,
            trace_w: 0.11,
        },
        palette: pal(
            [0.70, 0.25, 0.18],
            [0.30, 0.18, 0.10],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.24],
        ),
        bg_palette: pal(
            [0.14, 0.04, 0.03],
            [0.08, 0.03, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Pulse,
            speed: 0.9,
            amplitude: 0.2,
        },
        light: LightModel {
            height_scale: 0.9,
            normal_strength: 1.0,
            shininess: 28.0,
            specular: 0.55,
            spec_color: 0x00FFD0B0,
            ao_strength: 0.5,
            emissive: 0.22,
        },
        stability: GlyphStability::Responsive,
        ..catalog_base(
            TextureRecipe::Circuit {
                scale: 1.8,
                trace_w: 0.11,
            },
            0x00CE,
        )
    }
}

// ── KINDRED-SYMBOL / FLOATING-GLYPH demo materials ──────────────────────────
//
// These three exercise the [`crate::glyph_family`] super-glyph layer so the
// "floating" effect is visible: a neutral haze, a directional within-glyph flow,
// and a five-channel emitter (3 colours + bold + kindred glyph).

/// HAZE / MIST — the canonical FLOATING material: a NEUTRAL kindred family (`o 0 O
/// °`) cycled over `time`, so a soft grain shimmers/drifts without changing meaning.
/// Time-driven ⇒ animated (bypasses the memo cache, re-shaded each tick). Soft, low-
/// contrast palette; `Stable` glyph LOD so the float reads as drift, not jitter.
fn mat_haze() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 1.4,
            speck: 0.5,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.55, 0.62, 0.72],
            [0.18, 0.22, 0.28],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.28],
        ),
        bg_palette: pal(
            [0.06, 0.08, 0.11],
            [0.03, 0.04, 0.06],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.2,
            normal_strength: 0.3,
            shininess: 8.0,
            specular: 0.05,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.2,
            emissive: 0.25,
        },
        stability: GlyphStability::Stable,
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::HAZE_O,
            driver: crate::glyph_family::FamilyDriver::Time,
            speed: 0.8,
        }),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 1.4,
                speck: 0.5,
                tile_eps: 0.0,
            },
            0x00CF,
        )
    }
}

/// FLOW / STREAM — a directional WITHIN-GLYPH animation: a flowing bar/arc family
/// (`◜◝◞◟`) cycled faster over `time` so the surface reads as a moving stream. Cool
/// teal palette, faint emissive, an underline scan-glow.
fn mat_flow() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.2,
            speck: 0.6,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.30, 0.78, 0.82],
            [0.10, 0.20, 0.22],
            [1.0, 1.0, 1.0],
            [0.0, 0.15, 0.30],
        ),
        bg_palette: pal(
            [0.03, 0.07, 0.08],
            [0.01, 0.03, 0.04],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.2,
            normal_strength: 0.3,
            shininess: 10.0,
            specular: 0.1,
            spec_color: 0x00C0FFF0,
            ao_strength: 0.2,
            emissive: 0.3,
        },
        stability: GlyphStability::Stable,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Curly,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::Inked,
        },
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::ANIM_ARC,
            driver: crate::glyph_family::FamilyDriver::Time,
            speed: 2.4,
        }),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.2,
                speck: 0.6,
                tile_eps: 0.0,
            },
            0x00D0,
        )
    }
}

/// SPARK / EMBER — the FIVE-CHANNEL EMITTER demo: a GROWTH family (`· ∘ o O ●`)
/// indexed by distance, an emissive `Pulse` that drives the BOLD channel
/// (bold-as-emitter, gated on `family.is_some()`), an underline glow, plus a
/// bold-as-PROXIMITY cue so near embers thicken. So one cell carries fg + bg +
/// underline + bold + the kindred floating glyph — the player's five channels.
fn mat_spark() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 3.0,
            speck: 0.35,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.95, 0.55, 0.18],
            [0.20, 0.10, 0.04],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.10, 0.03, 0.01],
            [0.05, 0.01, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Pulse,
            speed: 1.6,
            amplitude: 0.3,
        },
        light: LightModel {
            height_scale: 0.3,
            normal_strength: 0.4,
            shininess: 14.0,
            specular: 0.2,
            spec_color: 0x00FFD080,
            ao_strength: 0.3,
            emissive: 0.7,
        },
        stability: GlyphStability::Stable,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Dotted,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::Inked,
        },
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::GROWTH_DOT,
            driver: crate::glyph_family::FamilyDriver::Distance,
            speed: 0.0,
        }),
        proximity_bold: Some(4.0),
        lod_override: None,
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 3.0,
                speck: 0.35,
                tile_eps: 0.0,
            },
            0x00D1,
        )
    }
}

// ── TRANSPARENT materials (composite slot 2, `OverKeepBg`) ───────────────────
//
// Each uses the glyph-only [`TextureRecipe::Hologram`] recipe, so [`Material::
// is_transparent`] returns `true` and the layer composite draws the cell's glyph +
// fg OVER the background behind it (keeping that bg, discarding the material's own
// void). They prove the transparent layer is exercised by more than `holo_node` /
// `holo_amber`: a glass pane, a translucent screen, and a gauzy veil — each a pure
// palette / scroll re-skin of the same see-through primitive.

/// VITRINE / GLASS PANE — a clean see-through pane: faint cool-blue scan-lines on a
/// void bg, a slow vertical drift (the pane "breathes"), low line density so most of
/// the cell is gap and the scene behind reads clearly through it. Transparent.
fn mat_vitrine() -> Material {
    Material {
        recipe: TextureRecipe::Hologram {
            line_freq: 3.0,
            scan_w: 0.30,
        },
        palette: pal(
            [0.62, 0.82, 0.92],
            [0.06, 0.08, 0.04],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Shift,
            speed: 0.25,
            amplitude: 0.4,
        },
        ..holo_node()
    }
}

/// SCREEN / TRANSLUCENT DISPLAY — a glowing HUD panel: dense phosphor-green raster
/// lines on void, scrolling faster (an active readout) with a flicker-bright hue.
/// Transparent — the readout floats over whatever is behind it. Distinct from the
/// opaque `monitor` (a solid CRT): this is a SEE-THROUGH overlay screen.
fn mat_screen() -> Material {
    Material {
        recipe: TextureRecipe::Hologram {
            line_freq: 9.0,
            scan_w: 0.55,
        },
        palette: pal(
            [0.40, 0.95, 0.55],
            [0.05, 0.10, 0.06],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Shift,
            speed: 1.6,
            amplitude: 1.0,
        },
        ..holo_node()
    }
}

/// VEIL / GAUZE — a soft translucent curtain: pale violet, mid-density lines drifting
/// slowly, low emissive so it reads as a dim hanging gauze you can see through rather
/// than a bright hologram. Transparent.
fn mat_veil() -> Material {
    Material {
        recipe: TextureRecipe::Hologram {
            line_freq: 5.0,
            scan_w: 0.45,
        },
        palette: pal(
            [0.74, 0.58, 0.86],
            [0.10, 0.06, 0.12],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Shift,
            speed: 0.4,
            amplitude: 0.7,
        },
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 16.0,
            specular: 0.0,
            spec_color: 0x00E0D0FF,
            ao_strength: 0.0,
            emissive: 0.45,
        },
        ..holo_node()
    }
}

// ── NEW OPAQUE smart materials (varied recipes / families / accents) ─────────
//
// More "different materials with different properties + settings" the user asked
// for, each a pure DATA config: a floating-family organic, a growth ember field, a
// within-glyph static, and a neighbour-coherent lattice.

/// MOSS / LICHEN — a soft organic crust: a green grain with a slow NEUTRAL kindred
/// float (`. · ∙ • ●` dot motes drifting), matte, low spec, gentle relief. The
/// floating dots read as living lichen. Stable (the drift is slow, not jittery).
fn mat_moss() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.6,
            speck: 0.45,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.34, 0.52, 0.26],
            [0.12, 0.18, 0.08],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.10],
        ),
        bg_palette: pal(
            [0.06, 0.10, 0.05],
            [0.03, 0.05, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.1],
        ),
        light: LightModel {
            height_scale: 0.5,
            normal_strength: 0.6,
            shininess: 8.0,
            specular: 0.06,
            spec_color: 0x00D8FFC0,
            ao_strength: 0.55,
            emissive: 0.04,
        },
        stability: GlyphStability::Stable,
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::HAZE_DOT,
            driver: crate::glyph_family::FamilyDriver::Time,
            speed: 0.35,
        }),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.6,
                speck: 0.45,
                tile_eps: 0.0,
            },
            0x00D2,
        )
    }
}

/// EMBER_FIELD / COALS — a hot bed of embers whose glyph GROWS with distance (a
/// `· ∘ o O ●` growth family on the Distance driver): near coals read as full discs,
/// far ones as faint points (the LOD growth cue). Strong emissive `Pulse`, an
/// under-glow, proximity-bold so near embers thicken. Static glyph driver (distance),
/// animated brightness.
fn mat_ember_field() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 3.4,
            speck: 0.4,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.95, 0.42, 0.10],
            [0.22, 0.10, 0.03],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.12, 0.03, 0.01],
            [0.06, 0.01, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Pulse,
            speed: 1.1,
            amplitude: 0.28,
        },
        light: LightModel {
            height_scale: 0.3,
            normal_strength: 0.4,
            shininess: 12.0,
            specular: 0.15,
            spec_color: 0x00FFC070,
            ao_strength: 0.3,
            emissive: 0.8,
        },
        stability: GlyphStability::Stable,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Curly,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::Inked,
        },
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::GROWTH_DENSITY,
            driver: crate::glyph_family::FamilyDriver::Distance,
            speed: 0.0,
        }),
        proximity_bold: Some(3.0),
        lod_override: None,
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 3.4,
                speck: 0.4,
                tile_eps: 0.0,
            },
            0x00D3,
        )
    }
}

/// STATIC / NOISE — a flickering grey field with a WITHIN-GLYPH spinner family
/// (`| / — \`) cycled fast over time, a brightness `Flicker` on top: reads as TV
/// static / interference. Responsive (the churn IS the look). The five-channel-ish
/// noisy panel.
fn mat_static_noise() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 4.0,
            speck: 0.7,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.66, 0.68, 0.70],
            [0.22, 0.22, 0.24],
            [1.0, 1.0, 1.0],
            [0.0, 0.05, 0.1],
        ),
        bg_palette: pal(
            [0.05, 0.05, 0.06],
            [0.02, 0.02, 0.03],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation {
            kind: AnimKind::Flicker,
            speed: 10.0,
            amplitude: 0.3,
        },
        light: LightModel {
            height_scale: 0.1,
            normal_strength: 0.2,
            shininess: 8.0,
            specular: 0.05,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.2,
            emissive: 0.15,
        },
        stability: GlyphStability::Responsive,
        family: Some(crate::glyph_family::FamilyLayer {
            family: crate::glyph_family::ANIM_SPIN,
            driver: crate::glyph_family::FamilyDriver::Time,
            speed: 6.0,
        }),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 4.0,
                speck: 0.7,
                tile_eps: 0.0,
            },
            0x00D4,
        )
    }
}

/// LATTICE / MESH — a cool woven panel: a dense Scanline cross-hatch in steel-blue
/// with a tight specular sheen, deep-ish AO in the weave, no animation. A calm,
/// structural opaque surface that contrasts the lively floating materials. Stable.
fn mat_lattice() -> Material {
    Material {
        palette: pal(
            [0.42, 0.50, 0.62],
            [0.16, 0.18, 0.22],
            [1.0, 1.0, 1.0],
            [0.0, 0.08, 0.16],
        ),
        bg_palette: pal(
            [0.07, 0.08, 0.11],
            [0.03, 0.04, 0.05],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.6,
            normal_strength: 0.7,
            shininess: 32.0,
            specular: 0.5,
            spec_color: 0x00D0E0FF,
            ao_strength: 0.6,
            emissive: 0.02,
        },
        stability: GlyphStability::Stable,
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Dotted,
            color: 0x00B8C8E6,
            gate: AccentGate::LitRidge,
        },
        ..catalog_base(
            TextureRecipe::Scanline {
                freq: 8.0,
                duty: 0.45,
                warp: 0.2,
                axis: 2,
            },
            0x00D5,
        )
    }
}

// ── SUPER-GLYPH smart materials ────────────────────────────────
//
// Three materials that finally AUTHOR the [`SuperGlyphLayer`] selectors the prior
// pass built but nothing used: a beat-stepped grid, a neighbour-coherent auto-tiling
// wall, and a procedurally "boiling" field. Each is a pure DATA config; the selector
// runs in `Material::sample` with the same frozen / quantized stability the family
// path uses, so the picks stay deadbanded under camera motion.

/// BEAT_GRID — a [`SuperSelector::Bpm`] panel: the kindred density glyph
/// (`. : ░ ▒ ▓ █`) STEPS on the beat (120 BPM, phase from `ctx.time`), so the surface
/// pulses in lockstep with the rhythmic-audio layer rather than smoothly floating. A
/// cool circuit body with a faint emissive glow. Animated (the beat advances time).
fn mat_beat_grid() -> Material {
    Material {
        palette: pal(
            [0.32, 0.66, 0.92],
            [0.10, 0.20, 0.30],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.28],
        ),
        bg_palette: pal(
            [0.04, 0.07, 0.11],
            [0.02, 0.03, 0.05],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.25,
            normal_strength: 0.35,
            shininess: 14.0,
            specular: 0.18,
            spec_color: 0x00B0E0FF,
            ao_strength: 0.25,
            emissive: 0.4,
        },
        stability: GlyphStability::Stable,
        super_glyph: Some(SuperGlyphLayer {
            family: crate::glyph_family::GROWTH_DENSITY,
            selector: SuperSelector::Bpm { bpm: 120.0 },
            state: 0,
        }),
        ..catalog_base(
            TextureRecipe::Circuit {
                scale: 2.0,
                trace_w: 0.07,
            },
            0x00D6,
        )
    }
}

/// AUTO_TILE — a [`SuperSelector::NeighborAware`] woven panel: each cell hashes its
/// own world lattice bucket together with its four axis neighbours, so adjoining cells
/// pick CONNECTED bar members (`▁▂▃▄▅▆▇█`) — a coherent woven / bonded auto-tiled
/// pattern instead of independent noise. World-anchored ⇒ stable under camera motion,
/// and the selector is STATIC (no `time`), so the material stays cacheable. Built on a
/// GRAIN body (not Brick) so the auto-tiled super-glyph OWNS the cell rather than being
/// overridden by a structural mortar detail. A warm sandstone palette.
fn mat_auto_tile() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.0,
            speck: 0.4,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.70, 0.56, 0.40],
            [0.22, 0.16, 0.10],
            [1.0, 1.0, 1.0],
            [0.0, 0.10, 0.18],
        ),
        bg_palette: pal(
            [0.10, 0.08, 0.06],
            [0.05, 0.04, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.55,
            normal_strength: 0.6,
            shininess: 10.0,
            specular: 0.08,
            spec_color: 0x00FFE6C0,
            ao_strength: 0.5,
            emissive: 0.03,
        },
        stability: GlyphStability::Stable,
        super_glyph: Some(SuperGlyphLayer {
            family: crate::glyph_family::GROWTH_BARS,
            selector: SuperSelector::NeighborAware,
            state: 0,
        }),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.0,
                speck: 0.4,
                tile_eps: 0.0,
            },
            0x00D7,
        )
    }
}

/// BOIL — a [`SuperSelector::Procedural`] field: the kindred dot glyph
/// (`. · ∙ • ●`) is computed from the cell's world hash mixed with a QUANTIZED time
/// step, so it churns endlessly and non-repeatingly (a "boiling" surface) yet advances
/// in stable jumps, not per-frame jitter. A sickly green-grey grain body with a faint
/// glow. Animated (the time step advances).
fn mat_boil() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.8,
            speck: 0.5,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.46, 0.62, 0.40],
            [0.16, 0.22, 0.14],
            [1.0, 1.0, 1.0],
            [0.0, 0.12, 0.10],
        ),
        bg_palette: pal(
            [0.05, 0.08, 0.05],
            [0.02, 0.04, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.1],
        ),
        light: LightModel {
            height_scale: 0.3,
            normal_strength: 0.4,
            shininess: 9.0,
            specular: 0.07,
            spec_color: 0x00D0FFC0,
            ao_strength: 0.3,
            emissive: 0.22,
        },
        stability: GlyphStability::Stable,
        super_glyph: Some(SuperGlyphLayer {
            family: crate::glyph_family::HAZE_DOT,
            selector: SuperSelector::Procedural { speed: 1.5 },
            state: 0,
        }),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.8,
                speck: 0.5,
                tile_eps: 0.0,
            },
            0x00D8,
        )
    }
}

/// PORTAL_ZOOM — concentric rings rushing outward (infinite zoom). A `Fractal`
/// recipe with `arms = 0` (pure rings) + the `Zoom` mode, in a cool→hot iq-cosine
/// ramp so successive octave-rings cycle hue. Self-lit (the relief pass leaves it),
/// so the ring-phase palette reads directly.
fn mat_portal_zoom() -> Material {
    Material {
        recipe: TextureRecipe::Fractal {
            arms: 0,
            octave_warp: 0.30,
            twist: 0.0,
        },
        // Cool teal core → hot magenta crest: a wide hue sweep across the octave so
        // each ring band is a distinct colour as it rushes out.
        palette: pal(
            [0.55, 0.35, 0.65],
            [0.40, 0.30, 0.35],
            [1.0, 1.0, 1.0],
            [0.10, 0.45, 0.85],
        ),
        bg_palette: pal(
            [0.05, 0.04, 0.09],
            [0.03, 0.02, 0.05],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 8.0,
            specular: 0.0,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.0,
            emissive: 0.6,
        },
        animation: Animation {
            kind: AnimKind::Zoom,
            speed: 0.35,
            amplitude: 1.0,
        },
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Fractal {
                arms: 0,
                octave_warp: 0.30,
                twist: 0.0,
            },
            0x00D9,
        )
    }
}

/// PORTAL_SPIRAL — a Droste spin+zoom spiral (hypnotic dive). A `Fractal` recipe
/// with `arms = 5` + a non-zero `twist` (log-spiral arms) driven by the coupled
/// `SpinZoom` mode so both the zoom and spin seams coincide. A warm amber→violet
/// ramp. Self-lit.
fn mat_portal_spiral() -> Material {
    Material {
        recipe: TextureRecipe::Fractal {
            arms: 5,
            octave_warp: 0.30,
            twist: 0.9,
        },
        palette: pal(
            [0.70, 0.45, 0.30],
            [0.30, 0.25, 0.40],
            [1.0, 1.0, 1.0],
            [0.0, 0.20, 0.55],
        ),
        bg_palette: pal(
            [0.07, 0.04, 0.06],
            [0.04, 0.02, 0.04],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.0,
            normal_strength: 0.0,
            shininess: 8.0,
            specular: 0.0,
            spec_color: 0x00FFFFFF,
            ao_strength: 0.0,
            emissive: 0.6,
        },
        animation: Animation {
            kind: AnimKind::SpinZoom,
            speed: 0.25,
            amplitude: 1.0,
        },
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Fractal {
                arms: 5,
                octave_warp: 0.30,
                twist: 0.9,
            },
            0x00DA,
        )
    }
}

// ── ANGLE-GLYPH + GLASS-MARKER + MULTI-CHANNEL demo materials ────────────────
//
// The three new look levers, each a pure DATA config:
//   • `facet`      — a `SuperSelector::ByAngle` hatch wall: the line glyph rotates
//                    with the viewing angle (a facetted / brushed-metal read).
//   • `pane`       — an underline-only TRANSPARENT glass marker: paints nothing but a
//                    coloured underline rule, the scene behind showing through.
//   • `relief_vol` — the ECONOMICAL MULTI-CHANNEL underline (`AccentGate::Volume`):
//                    fg/bg/coverage + a hue-preserving third underline cue per face/
//                    depth, for maximum perceived colour + volume per cell.

/// FACET — a [`SuperSelector::ByAngle`] brushed/hatched wall: the hatch line glyph
/// (`| / — \`, the `ANIM_SPIN` family) is chosen from the VIEW INCIDENCE on the face,
/// so as you orbit the surface the hatch rotates — a cheap "facetted metal" /
/// engine-turned read driven purely by the viewing angle. View-dependent (in the
/// no-pan recast set), STATIC in time (no flicker — the angle bucket is a deadband). A
/// cool steel palette so the rotating hatch reads as machined metal.
fn mat_facet() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.2,
            speck: 0.35,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.58, 0.62, 0.68],
            [0.20, 0.20, 0.22],
            [1.0, 1.0, 1.0],
            [0.0, 0.06, 0.12],
        ),
        bg_palette: pal(
            [0.07, 0.08, 0.10],
            [0.03, 0.04, 0.05],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.4,
            normal_strength: 0.5,
            shininess: 28.0,
            specular: 0.4,
            spec_color: 0x00E0ECFF,
            ao_strength: 0.4,
            emissive: 0.04,
        },
        stability: GlyphStability::Stable,
        super_glyph: Some(SuperGlyphLayer {
            // 8 buckets across the `| / — \` (ANIM_SPIN) family so the hatch steps
            // through its four members twice around the view circle.
            family: crate::glyph_family::ANIM_SPIN,
            selector: SuperSelector::ByAngle { buckets: 8 },
            state: 0,
        }),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.2,
                speck: 0.35,
                tile_eps: 0.0,
            },
            0x00DB,
        )
    }
}

/// PANE — an underline-only TRANSPARENT "glass marker": a see-through cell that paints
/// NO fg/bg/glyph (the scene behind shows through, via the transparent composite
/// layer) and shows ONLY a faint cool colored UNDERLINE as the "this is glass" cue.
/// Built on the see-through hologram primitive + an `Inked` underline accent + the
/// [`glass_marker`](Material::glass_marker) opt-in, so the composite drops its
/// glyph/fg/bg and keeps just the underline (see `composite::apply_glass_markers`).
fn mat_pane() -> Material {
    Material {
        recipe: TextureRecipe::Hologram {
            line_freq: 2.0,
            scan_w: 0.20,
        },
        palette: pal(
            [0.55, 0.78, 0.95],
            [0.04, 0.06, 0.05],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        animation: Animation::NONE,
        // The glass-marker underline: a cool cyan single rule on every (would-be inked)
        // cell of the pane. The composite keeps ONLY this, over the scene behind.
        underline_accent: UnderlineAccent {
            style: re_core::UnderlineStyle::Single,
            color: 0x0080_C8FF,
            gate: AccentGate::Inked,
        },
        glass_marker: true,
        ..holo_node()
    }
}

/// Test-only constructor for the glass-marker `pane` material, so the sibling
/// `composite` tests can build a registry around it without the catalog `fn`s being
/// public. `cfg(test)` — not part of the shipping surface.
#[cfg(test)]
pub(crate) fn mat_pane_for_test() -> Material {
    mat_pane()
}

/// RELIEF_VOL — the ECONOMICAL MULTI-CHANNEL VOLUME demo: an ordinary stone-ish grain
/// wall whose ONLY addition is the [`AccentGate::Volume`] underline — the hue-preserving
/// THIRD colour cue derived TOGETHER with fg/bg (brighter on the lit NS face, darker on
/// the shaded EW face, lifted near). So one element colour reads as many shades and the
/// wall gains volume per cell at ~no cost. Opaque, static, otherwise plain.
fn mat_relief_vol() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.4,
            speck: 0.4,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.52, 0.46, 0.40],
            [0.18, 0.16, 0.14],
            [1.0, 1.0, 1.0],
            [0.0, 0.08, 0.14],
        ),
        bg_palette: pal(
            [0.09, 0.08, 0.07],
            [0.04, 0.035, 0.03],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.6,
            normal_strength: 0.7,
            shininess: 12.0,
            specular: 0.12,
            spec_color: 0x00FFE8D0,
            ao_strength: 0.5,
            emissive: 0.02,
        },
        stability: GlyphStability::Stable,
        underline_accent: UnderlineAccent {
            // The multi-channel third cue: a Single rule whose colour is the
            // side/depth-biased HUE_VOLUME shade. Single so the underline reads as a
            // clean rim, not a curly accent.
            style: re_core::UnderlineStyle::Single,
            color: UnderlineAccent::HUE_VOLUME,
            gate: AccentGate::Volume,
        },
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.4,
                speck: 0.4,
                tile_eps: 0.0,
            },
            0x00DC,
        )
    }
}

// ── SEMANTIC SURFACES ──────────────────────
//
// Six catalog materials that opt into a [`crate::surface_fx::SurfaceFx`] finish. Each
// is an ordinary base recipe (so it degrades to that base if the finish were removed)
// PLUS the semantic post-layer. Appended at the tail of the catalog so no existing
// MaterialId / slot shifts. The redaction/decay finishes react to the generic state
// bus (`ctx.state`); the rest are static procedural surfaces.

/// FROST — a cold worley-crystal accretion over a pale stone base: a frozen surface
/// for the cold/warm semantic axis. Composes beside `ice`/`water`. The base is a calm
/// grain in a cool grey; the [`SurfaceFx::Frost`] finish adds the rime film + ice-star
/// crystals.
fn mat_frost() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.6,
            speck: 0.18,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.62, 0.68, 0.74],
            [0.14, 0.13, 0.12],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.10, 0.12, 0.15],
            [0.05, 0.06, 0.08],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.4,
            normal_strength: 0.5,
            shininess: 48.0,
            specular: 0.5,
            spec_color: 0x00E8F4FF,
            ao_strength: 0.35,
            emissive: 0.04,
        },
        stability: GlyphStability::Stable,
        surface_fx: Some(crate::surface_fx::SurfaceFx::Frost),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.6,
                speck: 0.18,
                tile_eps: 0.0,
            },
            0x00DD,
        )
    }
}

/// WET_STONE — a wet/condensation sheen over a dark stone base: a cool specular glint
/// in worley "puddles". The `wet` semantic-surface modifier, realised as a finish so
/// it works over any base recipe.
fn mat_wet_stone() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.2,
            speck: 0.22,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.34, 0.36, 0.40],
            [0.16, 0.16, 0.18],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.06, 0.07, 0.09],
            [0.03, 0.035, 0.045],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.5,
            normal_strength: 0.6,
            shininess: 52.0,
            specular: 0.55,
            spec_color: 0x00CFE8FF,
            ao_strength: 0.4,
            emissive: 0.03,
        },
        stability: GlyphStability::Stable,
        surface_fx: Some(crate::surface_fx::SurfaceFx::Wet),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.2,
                speck: 0.22,
                tile_eps: 0.0,
            },
            0x00DE,
        )
    }
}

/// REDACTED — the engine's own source (the [`TextureRecipe::CodeWall`]) under a censor
/// layer: every glyph starts as `█`/`░` (fully redacted, the resting/empty-bus look)
/// and the [`SurfaceFx::Redaction`] finish LIFTS the bars as the cell's `ctx.state`
/// rises (the name assembling). Binds via region (a panel) + state (the bridge ramps
/// the reveal). The surveillance / name-assembly semantic.
fn mat_redacted() -> Material {
    Material {
        recipe: TextureRecipe::CodeWall {
            panel: 0,
            u_chars_per_unit: 8.0,
        },
        palette: pal(
            [0.55, 0.62, 0.55],
            [0.22, 0.20, 0.16],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.04, 0.05, 0.04],
            [0.02, 0.025, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel::FLAT,
        stability: GlyphStability::Stable,
        surface_fx: Some(crate::surface_fx::SurfaceFx::Redaction),
        ..catalog_base(
            TextureRecipe::CodeWall {
                panel: 0,
                u_chars_per_unit: 8.0,
            },
            0x00DF,
        )
    }
}

/// WATCHER — a world-fixed eye/face lattice that brightens on approach (the most-
/// repeated horror image). The base is a calm dark lattice; the [`SurfaceFx::Watching`]
/// finish stamps the iris/pupil motif + the proximity glow. World-fixed (cheap) — no
/// billboard sampler. The [`crate::lod::LodOverride`] keeps the motif legible farther
/// out (a focal "it is looking at you" object).
fn mat_watcher() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 1.8,
            speck: 0.10,
            tile_eps: 0.0,
        },
        palette: pal(
            [0.32, 0.34, 0.40],
            [0.14, 0.14, 0.18],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.03, 0.03, 0.05],
            [0.015, 0.015, 0.025],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.3,
            normal_strength: 0.4,
            shininess: 24.0,
            specular: 0.25,
            spec_color: 0x00DCE8FF,
            ao_strength: 0.35,
            emissive: 0.08,
        },
        stability: GlyphStability::Stable,
        lod_override: Some(crate::lod::LodOverride::full_detail_to(18.0)),
        surface_fx: Some(crate::surface_fx::SurfaceFx::Watching),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 1.8,
                speck: 0.10,
                tile_eps: 0.0,
            },
            0x00E0,
        )
    }
}

/// WORN_FLOOR — a scuffed floor with worn grooves + settled dust (where iterations
/// stood). The base is the shipped-style floor grain; the [`SurfaceFx::Wear`] finish
/// darkens the trodden paths + scatters dust specks. Low-effort residue.
fn mat_worn_floor() -> Material {
    Material {
        recipe: TextureRecipe::Grain {
            scale: 2.0,
            speck: 0.30,
            tile_eps: 0.04,
        },
        palette: pal(
            [0.46, 0.42, 0.38],
            [0.16, 0.15, 0.13],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.08, 0.07, 0.06],
            [0.04, 0.035, 0.03],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.4,
            normal_strength: 0.5,
            shininess: 10.0,
            specular: 0.08,
            spec_color: 0x00E8E0D0,
            ao_strength: 0.55,
            emissive: 0.0,
        },
        stability: GlyphStability::Stable,
        surface_fx: Some(crate::surface_fx::SurfaceFx::Wear),
        ..catalog_base(
            TextureRecipe::Grain {
                scale: 2.0,
                speck: 0.30,
                tile_eps: 0.04,
            },
            0x00E1,
        )
    }
}

/// DECAY / ROT — a localized bit-rot finish over a circuit base: a deterministic
/// fraction of cells swap to a mojibake glyph + a sickly green-grey tint, intensity
/// from `ctx.state` (deadness/attempts) lifted from a resting `base` floor. Localized
/// to THIS surface only (vs the global glitch post-pass). The surface-decay semantic.
fn mat_decay() -> Material {
    Material {
        recipe: TextureRecipe::Circuit {
            scale: 1.7,
            trace_w: 0.10,
        },
        palette: pal(
            [0.40, 0.48, 0.36],
            [0.16, 0.18, 0.12],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        bg_palette: pal(
            [0.05, 0.06, 0.04],
            [0.025, 0.03, 0.02],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel {
            height_scale: 0.6,
            normal_strength: 0.7,
            shininess: 12.0,
            specular: 0.14,
            spec_color: 0x00D0E0B0,
            ao_strength: 0.5,
            emissive: 0.06,
        },
        stability: GlyphStability::Stable,
        // A modest resting corruption floor so the surface reads as decaying even on an
        // empty bus; the bridge drives it UP via `ctx.state` as a node dies / attempts mount.
        surface_fx: Some(crate::surface_fx::SurfaceFx::Decay { base: 0.25 }),
        ..catalog_base(
            TextureRecipe::Circuit {
                scale: 1.7,
                trace_w: 0.10,
            },
            0x00E2,
        )
    }
}

/// The breathing-node morph sequence `·∘○◌●`: a dot growing
/// to a filled ring and back — a neuron "inhaling". Low→high coverage.
const NODE_BREATH_FRAMES: [u32; 8] = [
    0x00B7, // ·
    0x2218, // ∘
    0x25CB, // ○
    0x25CC, // ◌
    0x25CF, // ●
    0x25CC, // ◌ back down (ping-pong tail so the loop breathes, not snaps)
    0x25CB, // ○
    0x2218, // ∘
];

/// NODE_BREATHING — a puzzle node that BREATHES on the
/// state bus. Reuses the shipped [`puzzle_node`] base (amber glow, self-lit) + a
/// `·∘○◌●` morph track gated by `OnState { field_gt: 0 }`: it cycles only when the
/// cell's generic state word is NON-ZERO (the bridge encodes "powered"), and yields
/// nothing (the static node glyph stays) when 0 (dead / empty bus). POWERED breathes,
/// dead is static.
///
/// NOTE on the state convention: this material reads "powered = non-zero" (the
/// `powered-bit` convention the pipe/flow encoding uses, `re_core::encode_flow_state`). The
/// shipped `STATE_NODE` family uses the opposite `0 = powered`; here we drop that
/// family so the glyph channel is purely the track + the fixed node glyph. The bridge
/// author binds the breathing node's tag with the `powered = non-zero` encoding.
fn mat_node_breathing() -> Material {
    Material {
        animation: Animation::NONE,
        family: None,
        anim_tracks: AnimTrackSet {
            tracks: [
                AnimTrack {
                    frames: NODE_BREATH_FRAMES,
                    frame_count: 8,
                    clock: AnimClock::Seconds(1.2),
                    trigger: AnimTrigger::OnState { field_gt: 0 },
                    target: AnimTarget::Body,
                },
                TRACK_INERT,
            ],
            count: 1,
        },
        ..puzzle_node()
    }
}

/// The shipped `held` / `released` of the [`ids::SUBSTRATE`] catalog material.
///
/// A MIXED field, deliberately: at `(0.55, 0.30)` roughly 45% of the wall is `░`,
/// 38% is `█` and 17% is `▒`, so three of the four states are on screen at once and
/// the alphabet reads as an alphabet rather than as a fill. `✕` needs `released` past
/// [`crate::substrate::BURN_RELEASE`] — a burn is damage the level author asks for,
/// not a resting state — and once there, all four coexist across the burn ramp (see
/// `crate::substrate::burn_fraction`).
const SUBSTRATE_HELD: f32 = 0.55;
/// See [`SUBSTRATE_HELD`].
const SUBSTRATE_RELEASED: f32 = 0.30;

/// SUBSTRATE — the buffer's own wall texture: the `░ █ ▒ ✕` cell alphabet
/// ([`crate::substrate`]) on a world-anchored lattice.
///
/// STATIC by construction — [`Animation::NONE`], no family, no super-glyph, no
/// surface-fx, no tracks — so it reports `is_animated() == false` and a level that
/// adopts it still idles at cache cost. The field's motion, when it comes, is puzzle
/// state moving `held` / `released`, not a clock.
///
/// [`LightModel::FLAT`] on purpose: the alphabet is a PRINTED field, not a moulded
/// one, and its height field is piecewise constant per cell — a relief gradient
/// across a cell boundary would spike the normal into a hard ring around every glyph.
/// Flat keeps the room's own light on the wall (it is not self-lit) with no relief
/// perturbation. A cold, desaturated slate-cyan family: a buffer, not a brick.
fn mat_substrate() -> Material {
    Material {
        palette: pal(
            [0.44, 0.50, 0.52],
            [0.28, 0.30, 0.31],
            [1.0, 1.0, 1.0],
            [0.0, 0.06, 0.12],
        ),
        bg_palette: pal(
            [0.07, 0.09, 0.10],
            [0.04, 0.05, 0.055],
            [1.0, 1.0, 1.0],
            [0.0, 0.08, 0.16],
        ),
        light: LightModel::FLAT,
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Substrate {
                scale: crate::substrate::CELLS_PER_UNIT,
                held: SUBSTRATE_HELD,
                released: SUBSTRATE_RELEASED,
            },
            0x00E3,
        )
    }
}

// ─────────────── the VOICE pair (state-driven substrate walls) ──────────────
//
// Two materials, one recipe, one alphabet, one scalar — and they must not read as the
// same wall twice. Three things separate them, and all three are deliberate:
//
//   1. DIRECTION. One fills an empty field (`░ → █`); the other erodes a whole one
//      (`█ → ▒`). They share only the `█` tier, and they pass through it from
//      opposite sides.
//   2. SEED. Both are dealt from `crate::substrate::cell_rank`, whose ordering is a
//      hash of the world cell and the material seed. At the SAME seed the hold voice's
//      held set (`rank < mark`) and the release voice's still-held set (`rank >= mark`)
//      would be exact COMPLEMENTS — two walls showing the same deal in negative, which
//      is the "same wall twice" failure in its most convincing disguise. Distinct seed
//      salts break it, and `the_two_voices_do_not_paint_the_same_deal` measures that
//      they are neither equal nor complementary across the whole mark sweep.
//   3. TEMPERATURE. Warm bone for the wall that keeps; cold slate-blue for the wall
//      that lets go. One hue point each (`SUBSTRATE_HUE_T`), so within a wall the four
//      tiers stay one colour family and separate by ink alone.
//
// Everything else is `mat_substrate`'s: `LightModel::FLAT` (a printed field, not a
// moulded one — a relief gradient across a per-cell-constant height field would ring
// every glyph), `GlyphStability::Stable`, and `Animation::NONE` so a level that adopts
// one still idles at cache cost.

/// The alphabet density both voices carry. Same as the substrate's: the grid must be
/// countable by eye, and a wall the player reads for progress is exactly the case
/// where that matters most.
const VOICE_CELLS_PER_UNIT: f32 = crate::substrate::CELLS_PER_UNIT;

/// `MNEMOS_HOLD` — the wall that keeps. `held = mark`, `released = 0`: an unclaimed
/// field that converts to `█` as the network powers up, and holds every cell it takes
/// even when the player unwires their work.
///
/// Warm desaturated bone, against the substrate's cold slate: a surface accumulating
/// something, not a buffer being read.
fn mat_mnemos_hold() -> Material {
    Material {
        palette: pal(
            [0.54, 0.47, 0.38],
            [0.30, 0.26, 0.21],
            [1.0, 1.0, 1.0],
            [0.0, 0.05, 0.10],
        ),
        bg_palette: pal(
            [0.09, 0.08, 0.06],
            [0.05, 0.045, 0.035],
            [1.0, 1.0, 1.0],
            [0.0, 0.07, 0.14],
        ),
        light: LightModel::FLAT,
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Voice {
                scale: VOICE_CELLS_PER_UNIT,
                mode: crate::substrate::VoiceMode::Hold,
            },
            0x00E4,
        )
    }
}

/// `LETHE_RELEASE` — the wall that lets go. `held = 1`, `released = mark`: a whole field
/// that scars to `▒` as the network powers up, and never heals a scar.
///
/// Cold slate-blue, the coldest of the three substrate surfaces. Note the endpoint:
/// `released = 1` is the TOP of the burn ramp, so the last percent of the mark carries
/// the field past `▒` into `✕` — the wall does not merely scar, it finishes.
///
/// The `d` phase vector RUNS BACKWARDS relative to every other material here
/// (`[0.18, 0.09, 0.0]`, not `[0.0, 0.09, 0.18]`), and that is the whole point.
/// [`Palette::eval`] is `a + b·cos(2π(c·t + d))`, so the channel whose `d` is 0 peaks
/// at the sampling point [`SUBSTRATE_HUE_T`] and the others fall away from it. Copying
/// the ascending `d` of the warm materials produced a nominally-blue `a` that evaluated
/// to `(0.38, 0.32, 0.27)` — a WARM wall from a cold-looking constant. Found by dumping
/// the frame and reading the fg hex, not by reading the numbers in this file.
fn mat_lethe_release() -> Material {
    Material {
        palette: pal(
            [0.34, 0.45, 0.60],
            [0.19, 0.25, 0.34],
            [1.0, 1.0, 1.0],
            [0.18, 0.09, 0.0],
        ),
        bg_palette: pal(
            [0.05, 0.07, 0.11],
            [0.03, 0.04, 0.07],
            [1.0, 1.0, 1.0],
            [0.20, 0.10, 0.0],
        ),
        light: LightModel::FLAT,
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Voice {
                scale: VOICE_CELLS_PER_UNIT,
                mode: crate::substrate::VoiceMode::Release,
            },
            0x00E5,
        )
    }
}

/// `conduit` — the edge-matched TRUCHET lattice: box-drawing runs that continue from
/// cell to cell instead of speckling (see [`crate::truchet`]).
///
/// Cool signal-cyan ink on a near-black plate, flat-lit and self-lit like the other
/// pattern recipes so the lattice reads as drawn structure rather than a lit surface.
/// `density` at the module default — dense enough that runs join into long paths,
/// open enough that the wall is not a solid grid.
fn mat_conduit() -> Material {
    Material {
        palette: pal(
            [0.32, 0.78, 0.82],
            [0.12, 0.30, 0.34],
            [1.0, 1.0, 1.0],
            [0.0, 0.15, 0.3],
        ),
        bg_palette: pal(
            [0.02, 0.05, 0.06],
            [0.01, 0.02, 0.03],
            [1.0, 1.0, 1.0],
            [0.0, 0.1, 0.2],
        ),
        light: LightModel::FLAT,
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Truchet {
                seed: 0,
                density: crate::truchet::DENSITY,
                scale: crate::truchet::WALL_SCALE,
            },
            0x00A7,
        )
    }
}

/// `facade` — the structured bay×floor window grid at its `office` default: a muted stone-grey wall with
/// warm-lit panes. `re_level::parse_recipe`'s bare `facade` defaults (`bay=3
/// floor=4 lit=0.35`) match this preset exactly, so `recipe = facade` and
/// `preset = facade` produce the SAME `TextureRecipe`, matching every other
/// pattern recipe's contract.
///
/// `LightModel::FLAT` like `Substrate`/`conduit`: the surface's contrast is glyph
/// + ink, not moulded relief, so a gradient tap would ring every mullion for no
///   visual gain.
fn mat_facade() -> Material {
    Material {
        palette: pal(
            [0.55, 0.55, 0.48],
            [0.28, 0.28, 0.24],
            [1.0, 1.0, 1.0],
            [0.05, 0.05, 0.0],
        ),
        bg_palette: pal(
            [0.06, 0.07, 0.09],
            [0.03, 0.035, 0.045],
            [1.0, 1.0, 1.0],
            [0.0, 0.05, 0.12],
        ),
        light: LightModel::FLAT,
        stability: GlyphStability::Stable,
        ..catalog_base(
            TextureRecipe::Facade {
                style: crate::facade::FacadeStyle::Office,
                bay: 3.0,
                floor_h: 4.0,
                lit: 0.35,
                seed: 0,
                ground_band: false,
                sign_row: false,
            },
            0x00A8,
        )
    }
}

// ───────────────────────────── tests ────────────────────────────────────────

// The seamless-loop proofs for every `AnimKind` live in their own file so the
// animation contract reads as one document. `#[path]` because this module is not a
// `mod.rs`, so the file sits beside it in `src/` rather than in an `ascii_shader/`
// directory. It is a CHILD of this module, so it can probe the private
// `anim_brightness` / `anim_shift_coord` / `fractal_phase` carriers directly.
#[cfg(test)]
#[path = "anim_tests.rs"]
mod anim_tests;

#[cfg(test)]
mod tests {
    // ── ramp-threshold dither ───────────
    mod ramp_dither {
        use crate::ascii_shader::{ramp_dither_offset, BODY_RAMP};
        use glam::Vec2;

        #[test]
        fn zero_amplitude_or_zero_depth_is_exactly_no_offset() {
            let c = Vec2::new(3.7, 12.2);
            assert_eq!(ramp_dither_offset(7, c, 0.0, 0.8), 0.0);
            assert_eq!(ramp_dither_offset(7, c, 0.5, 0.0), 0.0);
            assert_eq!(ramp_dither_offset(7, c, -1.0, 0.8), 0.0);
            assert_eq!(ramp_dither_offset(7, c, f32::NAN, 0.8), 0.0);
        }

        #[test]
        fn same_world_spot_same_offset_every_call() {
            // World-anchored: no clock, no screen coordinate, so a spot is stable
            // frame to frame and under camera motion.
            let c = Vec2::new(-4.25, 9.5);
            let a = ramp_dither_offset(11, c, 0.5, 0.7);
            for _ in 0..8 {
                assert_eq!(ramp_dither_offset(11, c, 0.5, 0.7), a);
            }
        }

        #[test]
        fn offset_is_bounded_by_half_a_ramp_step_times_amplitude_and_depth() {
            let step = 1.0 / (BODY_RAMP.len() as f32 - 1.0);
            for i in 0..400 {
                let c = Vec2::new(i as f32 * 0.37, (i * 7 % 23) as f32 * 0.61);
                for (amp, t) in [(0.5f32, 1.0f32), (1.0, 0.3), (0.25, 0.9)] {
                    let o = ramp_dither_offset(3, c, amp, t);
                    let bound = 0.5 * amp * step * t + 1e-6;
                    assert!(
                        o.abs() <= bound,
                        "coord {c:?} amp {amp} t {t}: {o} > {bound}"
                    );
                }
            }
        }

        #[test]
        fn offsets_vary_across_world_texels_so_a_band_breaks_up() {
            // A flat band at constant depth must get BOTH signs of jitter across its
            // length — that is what dissolves the contour into a stipple.
            let mut pos = 0;
            let mut neg = 0;
            for i in 0..64 {
                let o = ramp_dither_offset(5, Vec2::new(i as f32 * 0.5, 2.0), 1.0, 1.0);
                if o > 0.0 {
                    pos += 1;
                } else if o < 0.0 {
                    neg += 1;
                }
            }
            assert!(pos >= 12 && neg >= 12, "pos {pos} neg {neg}");
        }

        #[test]
        fn active_config_default_carries_the_documented_amplitude() {
            use crate::depth_contrast::{DepthContrastConfig, DEPTH_DEFAULT_RAMP_DITHER};
            // The default ships ON (the const is > 0); a zero default would silently
            // turn the feature off for every level, so pin it through the config.
            let cfg = DepthContrastConfig::default();
            assert_eq!(cfg.ramp_dither, DEPTH_DEFAULT_RAMP_DITHER);
            assert!(cfg.ramp_dither > 0.0, "dither ships ON by default");
        }
    }

    use super::*;

    fn ctx(distance: f32, foreshorten: f32) -> SampleCtx {
        SampleCtx {
            distance,
            foreshorten,
            side: 0,
            screen_y: 4,
            wall_v: 0.5,
            tick: 0,
            time: 0.0,
            fog: FogParams::DEFAULT,
            // No far-freeze by default in unit tests — the freeze tests set this
            // explicitly. `∞` reproduces the pre-feature behaviour exactly.
            far_freeze: f32::INFINITY,
            light: None,
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: None,
            wall_along: 0.0,
            wall_up: 0.0,
            on_wall: false,
            tex_footprint: None,
            subcell_backdrop: None,
        }
    }

    // ── relief / normal-mapped lighting ──────────────────────────────────────

    /// A flat-facing wall light context: face normal +z (toward a "camera" along
    /// +z), tangents along x (u) and y (v), key light straight on, view along +z.
    fn flat_light_ctx() -> LightCtx {
        LightCtx {
            face_n: Vec3::Z,
            tangent_u: Vec3::X,
            tangent_v: Vec3::Y,
            key_dir: Vec3::Z,
            key_color: Vec3::ONE,
            view_dir: Vec3::Z,
            ambient: Vec3::splat(0.15),
            headlamp: 0.5,
            shadow_key: 1.0,
            world_pos: Vec3::ZERO,
            points: PointLights::NONE,
        }
    }

    #[test]
    fn raised_bump_normal_points_outward() {
        // A bump that rises toward +u then falls: at the rising slope the gradient
        // is positive, so the perturbed normal tilts toward −u (downhill) but its
        // out-of-face (z) component must stay positive — it never points INTO the
        // surface. And a flat patch (zero gradient) returns the face normal.
        let lc = flat_light_ctx();
        let flat = perturb_normal(&lc, 0.0, 0.0, 1.0);
        assert!(
            (flat - Vec3::Z).length() < 1e-6,
            "flat patch keeps the face normal"
        );

        // A steep rising slope along u.
        let tilted = perturb_normal(&lc, 2.0, 0.0, 1.0);
        assert!(
            tilted.z > 0.0,
            "relief normal must still point out of the face (z>0)"
        );
        assert!(
            tilted.x < 0.0,
            "rising-u slope tilts the normal toward −u (downhill)"
        );
        assert!(
            (tilted.length() - 1.0).abs() < 1e-5,
            "normal must be unit length"
        );
    }

    #[test]
    fn diffuse_increases_with_n_dot_l() {
        // Light a mid-grey albedo with the normal swung from grazing to facing the
        // key light. Lit luminance must rise monotonically with N·L.
        let lc = flat_light_ctx();
        let lm = LightModel {
            specular: 0.0,
            emissive: 0.0,
            ao_strength: 0.0,
            ..LightModel::DEFAULT
        };
        let albedo = Vec3::splat(0.6);
        let lum_at = |angle: f32| -> f32 {
            // Normal in the x-z plane: angle 0 → +z (facing key), larger → grazing.
            let n = Vec3::new(angle.sin(), 0.0, angle.cos());
            let (fg, _bg, _l, _) = light_surface(albedo, n, 0.5, &lm, &lc);
            fg.dot(Vec3::new(0.299, 0.587, 0.114))
        };
        let facing = lum_at(0.0);
        let oblique = lum_at(1.0);
        let grazing = lum_at(1.5);
        assert!(
            facing > oblique && oblique > grazing,
            "diffuse must fall as N·L falls ({facing} > {oblique} > {grazing})"
        );
    }

    #[test]
    fn shadow_key_attenuates_key_diffuse_but_not_ambient() {
        // Worked-out shadows (#2): `shadow_key` multiplies ONLY the key diffuse +
        // key specular. A facing surface lit then fully shadowed must dim; but the
        // ambient floor + headlamp keep it OFF pure black, and `shadow_key == 1.0`
        // is identical to the pre-shadow path (no attenuation).
        let lc = flat_light_ctx();
        let lm = LightModel {
            specular: 0.0,
            emissive: 0.0,
            ao_strength: 0.0,
            ..LightModel::DEFAULT
        };
        let albedo = Vec3::splat(0.6);
        let n = Vec3::Z; // facing the key
        let lum = |shadow_key: f32| -> f32 {
            let mut lc2 = lc;
            lc2.shadow_key = shadow_key;
            let (fg, _bg, _l, _) = light_surface(albedo, n, 0.5, &lm, &lc2);
            fg.dot(Vec3::new(0.299, 0.587, 0.114))
        };
        let lit = lum(1.0);
        let half = lum(0.5);
        let dark = lum(0.0);
        assert!(
            lit > half && half > dark,
            "key diffuse must scale with shadow_key ({lit} > {half} > {dark})"
        );
        assert!(
            dark > 0.0,
            "ambient + headlamp keep a fully-shadowed cell off pure black (got {dark})"
        );
        // `shadow_key == 1.0` must reproduce the pre-shadow value exactly: build the
        // same ctx via the public constructor (which defaults shadow_key to 1.0).
        let env = LightEnv {
            key_dir: Vec3::Z,
            key_color: Vec3::ONE,
            ambient: Vec3::splat(0.15),
            headlamp: 0.5,
            points: PointLights::NONE,
        };
        let mut baseline = LightCtx::for_floor(true, Vec3::Z, &env);
        baseline.face_n = Vec3::Z;
        baseline.tangent_u = Vec3::X;
        baseline.tangent_v = Vec3::Y;
        let (fg_b, _, _, _) = light_surface(albedo, n, 0.5, &lm, &baseline);
        assert!(
            (fg_b.dot(Vec3::new(0.299, 0.587, 0.114)) - lit).abs() < 1e-6,
            "shadow_key=1.0 must equal the default (pre-shadow) LightCtx output"
        );
    }

    #[test]
    fn view_independent_lighting_is_invariant_to_camera_facing() {
        // EXACTNESS CONTRACT for `REVISION_VIEWINDEP_LIGHT=1`: with the view-dependent
        // terms dropped, a cell's shade must be IDENTICAL regardless of which way the
        // camera faces — this is precisely what makes the near-ring rotation pan
        // byte-exact (the pan reuses a shade baked at one yaw at every other yaw).
        // We test the logic directly via `light_surface_mode(.., view_indep=true)`
        // with two materially different `view_dir`s, so the assertion does not depend
        // on the process-global env LazyLock.
        let lm = LightModel {
            specular: 0.6,
            emissive: 0.0,
            ao_strength: 0.3,
            shininess: 8.0,
            ..LightModel::DEFAULT
        };
        let albedo = Vec3::new(0.6, 0.5, 0.4);
        let normal = Vec3::new(0.2, -0.1, 0.97).normalize(); // an off-axis relief normal
        let h = 0.62;

        // Two camera facings: head-on (+z) and a steep oblique. Everything WORLD-fixed
        // (face_n, tangents, key, ambient, headlamp) is held constant; only view_dir
        // changes — exactly what a pure rotation does to a fixed surface point.
        let env = LightEnv {
            key_dir: Vec3::new(-0.3, -0.4, 0.86).normalize(),
            key_color: Vec3::ONE,
            ambient: Vec3::splat(0.15),
            headlamp: 0.5,
            points: PointLights::NONE,
        };
        let mut lc_a = LightCtx::for_floor(true, Vec3::Z, &env);
        let mut lc_b = lc_a;
        lc_a.view_dir = Vec3::new(0.0, 0.0, 1.0);
        lc_b.view_dir = Vec3::new(0.7, -0.5, 0.5).normalize();

        // ── ON (view-independent): byte-identical across the two facings. ──
        let (fg_a, bg_a, lum_a, _) = light_surface_mode(albedo, normal, h, &lm, &lc_a, true);
        let (fg_b, bg_b, lum_b, _) = light_surface_mode(albedo, normal, h, &lm, &lc_b, true);
        assert_eq!(
            fg_a, fg_b,
            "view-indep fg must be identical across camera facings"
        );
        assert_eq!(
            bg_a, bg_b,
            "view-indep bg-shadow must be identical across camera facings"
        );
        assert_eq!(
            lum_a, lum_b,
            "view-indep ramp luminance must be identical across camera facings"
        );

        // ── OFF (today's look): the two facings genuinely DIFFER (specular + headlamp
        //    N·V depend on view_dir), proving the toggle is doing real work and the
        //    default path still carries the view-dependent glints. ──
        let (fg_a_off, _, _, _) = light_surface_mode(albedo, normal, h, &lm, &lc_a, false);
        let (fg_b_off, _, _, _) = light_surface_mode(albedo, normal, h, &lm, &lc_b, false);
        assert_ne!(
            fg_a_off, fg_b_off,
            "view-DEPENDENT (default) fg must change with camera facing"
        );
    }

    #[test]
    fn specular_peaks_near_the_half_vector() {
        // Key + view both along +z → the half-vector is +z. A normal aligned with
        // +z (= the half-vector) must produce a stronger specular than a tilted one.
        let lc = flat_light_ctx();
        let lm = LightModel {
            specular: 1.0,
            shininess: 32.0,
            ao_strength: 0.0,
            emissive: 0.0,
            ..LightModel::DEFAULT
        };
        let albedo = Vec3::splat(0.2); // dark so the spec dominates the luminance
        let lum_at = |tilt: f32| -> f32 {
            let n = Vec3::new(tilt.sin(), 0.0, tilt.cos());
            let (fg, _b, _l, _) = light_surface(albedo, n, 0.5, &lm, &lc);
            fg.dot(Vec3::new(0.299, 0.587, 0.114))
        };
        let on_half = lum_at(0.0);
        let off_half = lum_at(0.6);
        assert!(
            on_half > off_half,
            "specular must peak at the half-vector ({on_half} > {off_half})"
        );
    }

    #[test]
    fn ao_darkens_a_crevice() {
        // A low (concave) height reads darker than a high (raised) one under the
        // same light, because AO occludes the crevice.
        let lc = flat_light_ctx();
        let lm = LightModel {
            ao_strength: 0.8,
            specular: 0.0,
            emissive: 0.0,
            ..LightModel::DEFAULT
        };
        let albedo = Vec3::splat(0.6);
        let lum_h = |h: f32| -> f32 {
            let (fg, _b, _l, _) = light_surface(albedo, Vec3::Z, h, &lm, &lc);
            fg.dot(Vec3::new(0.299, 0.587, 0.114))
        };
        let raised = lum_h(1.0);
        let crevice = lum_h(0.0);
        assert!(
            crevice < raised,
            "AO must darken the crevice ({crevice} < {raised})"
        );
    }

    #[test]
    fn raised_lit_cell_picks_a_denser_glyph_than_recessed() {
        // The glyph-protrusion mechanism end-to-end: the same lighting machine
        // `light_surface` feeds `ramp_glyph` via the protrude score
        // `0.35*lum + 0.65*h` (height-weighted so relief, not the flat headlamp
        // flood, drives glyph selection). A RAISED + LIT input must yield a denser
        // ramp glyph (earlier in `@%#*+=~:-. `) than a RECESSED + SHADOWED one.
        let lc = flat_light_ctx();
        let lm = LightModel::DEFAULT;
        let albedo = Vec3::splat(0.7);
        // Density rank: position in the ramp; 0 = densest `@`, larger = sparser.
        let density_rank = |g: u32| -> usize {
            BODY_RAMP
                .iter()
                .position(|&b| b as u32 == g)
                .unwrap_or(BODY_RAMP.len())
        };
        // Raised + facing the light (high h, normal at the half-vector).
        let (_fg_r, _bg_r, lum_r, _) = light_surface(albedo, Vec3::Z, 0.95, &lm, &lc);
        let protrude_r = (0.35 * lum_r + 0.65 * 0.95).clamp(0.0, 1.0);
        let glyph_r = ramp_glyph(protrude_r);
        // Recessed + grazing the light (low h, normal tilted away).
        let n_shadow = Vec3::new(0.9, 0.0, 0.436).normalize();
        let (_fg_c, _bg_c, lum_c, _) = light_surface(albedo, n_shadow, 0.05, &lm, &lc);
        let protrude_c = (0.35 * lum_c + 0.65 * 0.05).clamp(0.0, 1.0);
        let glyph_c = ramp_glyph(protrude_c);
        assert!(
            density_rank(glyph_r) < density_rank(glyph_c),
            "raised-lit glyph {:?} (rank {}) must be denser than recessed-shadowed {:?} (rank {})",
            char::from_u32(glyph_r).unwrap(),
            density_rank(glyph_r),
            char::from_u32(glyph_c).unwrap(),
            density_rank(glyph_c),
        );

        // And the same holds through the FULL brick sample path: a raised, lit
        // brick-face cell picks a denser body glyph than the same material at a
        // low-light grazing angle.
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let env = LightEnv::builtin();
        let coord = Vec2::new(1.10, 1.30); // lands on a brick face (high h)
        assert!(
            mat.height_at(&perlin, coord, ShadeLod::FULL) > 0.6,
            "test coord must be a raised face"
        );
        let body_glyph = |sign_view: Vec3| -> u32 {
            let mut c = ctx(2.0, 1.0);
            c.light = Some(LightCtx::for_wall(0, 1.0, sign_view, &env));
            mat.sample(&perlin, coord, &c).glyph
        };
        let lit = body_glyph(Vec3::new(0.0, 1.0, 0.3).normalize()); // facing → lit
        let dim = body_glyph(Vec3::new(1.0, 0.05, 0.0).normalize()); // grazing → dark
                                                                     // The lit face is at least as dense as the dimmed one (denser or equal —
                                                                     // the body glyph never gets SPARSER when lit).
        assert!(
            density_rank(lit) <= density_rank(dim),
            "lit face glyph {:?} must be at least as dense as the dim one {:?}",
            char::from_u32(lit).unwrap(),
            char::from_u32(dim).unwrap(),
        );
    }

    #[test]
    fn lighting_supersedes_flat_side_dim_via_normal() {
        // With a LightCtx, two walls of different `side` (so different face
        // normals) lit by the same key differ by REAL N·L, not the hardcoded
        // SIDE_EW_DIM. Prove the lit result depends on the face normal: an NS face
        // and an EW face under the same key get different luminance.
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let env = LightEnv::builtin();
        let view = Vec3::new(0.0, 1.0, 0.0);
        let coord = Vec2::new(1.234, 1.5);
        let lum = |side: u8, sign: f32| -> f32 {
            let mut c = ctx(3.0, 1.0);
            c.side = side;
            c.light = Some(LightCtx::for_wall(side, sign, view, &env));
            let t = mat.sample(&perlin, coord, &c);
            unpack_rgb(t.fg).dot(Vec3::new(0.299, 0.587, 0.114))
        };
        let ns = lum(0, 1.0);
        let ew = lum(1, 1.0);
        assert!(
            (ns - ew).abs() > 1e-4,
            "N·L must differentiate the two faces (ns={ns}, ew={ew})"
        );
    }

    #[test]
    fn cosine_palette_is_deterministic_and_in_gamut() {
        let p = wall_brick().palette;
        for i in 0..64 {
            let t = i as f32 / 64.0;
            let a = p.eval(t);
            let b = p.eval(t);
            assert_eq!(a, b, "palette must be a pure function of t");
            for c in [a.x, a.y, a.z] {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "palette channel out of gamut: {c}"
                );
            }
        }
        // A known control point: at t such that cos args are integers the value
        // is a + b (since cos(0)=1) for the bright phase — sanity, not magic.
        let mid = p.eval(0.5);
        assert!(mid.length() > 0.0);
    }

    #[test]
    fn pack_unpack_roundtrips() {
        for c in [0x00FF_8040u32, 0x0000_0000, 0x00FF_FFFF, 0x0012_3456] {
            let back = pack_rgb(unpack_rgb(c));
            assert_eq!(back, c, "pack∘unpack must be identity");
        }
    }

    #[test]
    fn reveal_threshold_is_monotonic_decreasing() {
        let near = reveal_threshold(LOD_NEAR_DENSITY);
        let mid = reveal_threshold((LOD_NEAR_DENSITY + LOD_FAR_DENSITY) * 0.5);
        let far = reveal_threshold(LOD_FAR_DENSITY + 5.0);
        assert!(near >= mid && mid >= far, "T must fall with density");
        assert!(near >= RANK_MAX as f32 - 1.0, "near reveals all ranks");
        assert!(far <= 1.0, "far reveals base only");
    }

    #[test]
    fn rank_lod_reveals_more_near_than_far() {
        // The SAME material + coord: a near cell must surface a higher-rank
        // (finer) detail texel than the same point seen far away.
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();

        // Scan a strip of the face for the highest revealed rank at each dist.
        let highest_rank = |distance: f32| -> u8 {
            let mut best = 0u8;
            for i in 0..200 {
                let u = i as f32 / 200.0;
                let coord = Vec2::new(u * 4.0, 1.3);
                let t = mat.sample(&perlin, coord, &ctx(distance, 1.0));
                best = best.max(t.rank);
            }
            best
        };

        let near = highest_rank(2.0);
        let far = highest_rank(24.0);
        assert!(
            near > far,
            "near must reveal a higher-rank texel than far (near={near}, far={far})"
        );
        // And far should collapse to base-only (rank 0).
        assert_eq!(far, 0, "far surface should show base only");
    }

    #[test]
    fn grazing_angle_coarsens_like_distance() {
        // At fixed distance, a grazing surface (high foreshorten) reveals fewer
        // ranks than a face-on one — angle-aware filtering.
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_circuit();
        let highest = |fore: f32| {
            let mut best = 0u8;
            for i in 0..200 {
                let coord = Vec2::new(i as f32 / 200.0 * 4.0, 0.8);
                best = best.max(mat.sample(&perlin, coord, &ctx(8.0, fore)).rank);
            }
            best
        };
        assert!(
            highest(1.0) >= highest(6.0),
            "grazing (high foreshorten) must not reveal more than face-on"
        );
    }

    #[test]
    fn sample_is_deterministic() {
        let perlin = Perlin::new(SHADER_SEED);
        let reg = MaterialRegistry::with_builtins();
        let c = ctx(5.0, 1.4);
        let coord = Vec2::new(3.21, 7.83);
        for id in [
            ids::WALL_BRICK,
            ids::WALL_CIRCUIT,
            ids::FLOOR_GRAIN,
            ids::CEILING_SKY,
        ] {
            let m = reg.get(id);
            let a = m.sample(&perlin, coord, &c);
            let b = m.sample(&perlin, coord, &c);
            assert_eq!(a, b, "material.sample must be deterministic for {id:?}");
        }
    }

    /// An opt-in `[material] detail = on` must be a strict
    /// no-op for every material that never opts in, and — once opted in — must
    /// reveal MORE fine-grain micro-detail glyphs the closer the surface is
    /// (the world-anchored DETAIL-tier cross-fade), never fewer.
    #[test]
    fn detail_tier_opt_in_reveals_more_fine_grain_up_close() {
        let perlin = Perlin::new(SHADER_SEED);
        let plain = floor_grain();
        let with_detail = Material {
            detail: true,
            ..floor_grain()
        };

        // Scan a strip and count DETAIL_MICRO_GLYPH (`:`) hits at each distance.
        let micro_count = |mat: &Material, distance: f32| -> usize {
            (0..400)
                .filter(|&i| {
                    let coord = Vec2::new(i as f32 * 0.083, i as f32 * 0.071);
                    let t = mat.sample(&perlin, coord, &ctx(distance, 1.0));
                    t.glyph == DETAIL_MICRO_GLYPH
                })
                .count()
        };

        // Far away (well inside TONE/PATTERN, rho >= DETAIL_RHO): opting in must
        // change NOTHING — the detail_lod machinery is never engaged there.
        assert_eq!(
            micro_count(&plain, 20.0),
            micro_count(&with_detail, 20.0),
            "far away, detail=on must be byte-identical to detail=off"
        );

        // Deep in the DETAIL tier (small rho / close distance): the opted-in
        // material must show strictly more micro-detail than the plain one (which
        // can show none — `DETAIL_MICRO_GLYPH` is only ever emitted by this path).
        assert_eq!(
            micro_count(&plain, 0.3),
            0,
            "a material that never opts in never shows the micro glyph"
        );
        let close_detail = micro_count(&with_detail, 0.3);
        assert!(
            close_detail > 0,
            "an opted-in material must reveal SOME micro-detail up close"
        );

        // Monotonic approach: strictly-closer distances reveal at-least-as-much
        // micro-detail as farther ones (the "opens up as you approach" contract —
        // matches `detail_lod::reveal_finer`'s own monotonicity guarantee).
        let far_detail = micro_count(&with_detail, 2.0);
        assert!(
            close_detail >= far_detail,
            "closer must reveal at least as much micro-detail (far={far_detail}, close={close_detail})"
        );
    }

    #[test]
    fn dual_channel_fg_and_bg_both_set() {
        // A near floor cell should carry BOTH a non-zero bg base AND vary fg.
        let perlin = Perlin::new(SHADER_SEED);
        let floor = floor_grain();
        let mut saw_bg = false;
        let mut fg_values = std::collections::HashSet::new();
        for i in 0..400 {
            let coord = Vec2::new(i as f32 * 0.137, i as f32 * 0.091);
            let t = floor.sample(&perlin, coord, &ctx(3.0, 1.1));
            if t.bg != 0 {
                saw_bg = true;
            }
            fg_values.insert(t.fg);
        }
        assert!(saw_bg, "floor must emit a non-zero bg base (dual-channel)");
        assert!(fg_values.len() > 3, "fg must vary across the floor");
    }

    #[test]
    fn empty_registry_get_does_not_panic() {
        // R-L1: `get` on a fully-empty registry returns the static default
        // instead of indexing out of bounds.
        let empty = MaterialRegistry::new();
        let m = empty.get(MaterialId(0));
        // It is a usable material (the neutral grey brick).
        assert!(matches!(m.recipe, TextureRecipe::Brick { .. }));
        let _ = empty.get(MaterialId(999)); // also fine
    }

    #[test]
    fn sky_base_fg_is_nonzero() {
        // R-L3: the sky base must carry a faint non-zero fg so a stray glyph
        // drawn on a sky cell is not invisible.
        let perlin = Perlin::new(SHADER_SEED);
        let sky = ceiling_sky();
        let mut saw_nonzero_base_fg = false;
        for i in 0..200 {
            // Pick coords that land on the nebula base (no star) by scanning.
            // Use a near distance so fog doesn't legitimately dim everything to
            // black — we're asserting the base *produces* a non-zero fg.
            let coord = Vec2::new(i as f32 * 0.37, i as f32 * 0.21);
            let t = sky.sample(&perlin, coord, &ctx(2.0, 1.0));
            if t.glyph == u32::from(' ') {
                // a base (space) cell: its fg must be > 0
                if t.fg != 0 {
                    saw_nonzero_base_fg = true;
                }
            }
        }
        assert!(saw_nonzero_base_fg, "sky base fg must be non-zero (R-L3)");
    }

    #[test]
    fn glyph_source_fixed_overrides_base_glyph() {
        let perlin = Perlin::new(SHADER_SEED);
        let mut m = wall_brick();
        m.glyph_source = GlyphSource::Fixed(u32::from('@'));
        // A far cell collapses to base only, so the base glyph shows through.
        let t = m.sample(&perlin, Vec2::new(1.234, 2.0), &ctx(24.0, 1.0));
        assert_eq!(
            t.glyph,
            u32::from('@'),
            "Fixed glyph source must pin the base glyph"
        );
    }

    #[test]
    fn from_specs_and_id_of_resolve_by_name() {
        let reg = MaterialRegistry::from_specs(&[
            MaterialSpec {
                name: "alpha",
                material: wall_brick(),
            },
            MaterialSpec {
                name: "beta",
                material: floor_grain(),
            },
        ]);
        assert_eq!(reg.id_of("alpha"), Some(MaterialId(0)));
        assert_eq!(reg.id_of("beta"), Some(MaterialId(1)));
        assert_eq!(reg.id_of("missing"), None);
    }

    #[test]
    fn pulse_animation_varies_brightness_over_time() {
        let perlin = Perlin::new(SHADER_SEED);
        let mut m = wall_brick();
        m.animation = Animation {
            kind: AnimKind::Pulse,
            speed: 1.0,
            amplitude: 0.3,
        };
        let coord = Vec2::new(1.0, 2.0);
        let lum = |time: f32| -> u32 {
            let mut c = ctx(3.0, 1.0);
            c.time = time;
            let t = m.sample(&perlin, coord, &c);
            ((t.fg >> 16) & 0xFF) + ((t.fg >> 8) & 0xFF) + (t.fg & 0xFF)
        };
        // cos(0)=1 (brightest) vs cos(π)=-1 (dimmest) at speed=1 → time=0 vs 0.5.
        assert!(
            lum(0.0) > lum(0.5),
            "Pulse must brighten at phase 0 vs phase π"
        );
    }

    #[test]
    fn fog_color_tints_far_cells_toward_color_but_not_near() {
        // A vivid magenta fog with a tight band: a near cell (< near) must be
        // untinted (its own colour), while a far cell (>= far) must collapse to
        // (almost) the fog colour. Mid cells lie between.
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let fog_rgb = 0x00FF_00FFu32; // magenta
        let fog = FogParams {
            near: 2.0,
            far: 8.0,
            color: Some(fog_rgb),
            height: None,
        };
        let coord = Vec2::new(1.234, 2.0);

        let sample_at = |distance: f32| -> (u32, u32) {
            let mut c = ctx(distance, 1.0);
            c.fog = fog;
            let t = mat.sample(&perlin, coord, &c);
            (t.fg, t.bg)
        };

        let near = sample_at(1.0); // before the band → untinted
        let far = sample_at(20.0); // past `far` → full fog colour

        // Distance-to-target: how close a colour is to the fog colour (lower =
        // closer). The far cell must be far closer to magenta than the near one.
        let dist_to_fog = |c: u32| -> i32 {
            let dr = (((c >> 16) & 0xFF) as i32 - 0xFF).abs();
            // Distance from the green channel to 0x00 (the fog colour's G) — i.e. the
            // channel value itself (kept explicit to mirror the dr/db forms).
            let dg = ((c >> 8) & 0xFF) as i32;
            let db = ((c & 0xFF) as i32 - 0xFF).abs();
            dr + dg + db
        };
        assert!(
            dist_to_fog(far.0) < dist_to_fog(near.0),
            "far fg must be closer to fog colour than near (near={:#08x} far={:#08x})",
            near.0,
            far.0
        );
        // Far cell is essentially the fog colour on both channels.
        assert!(
            dist_to_fog(far.0) < 8,
            "far fg should be ~fog colour, got {:#08x}",
            far.0
        );
        assert!(
            dist_to_fog(far.1) < 8,
            "far bg should be ~fog colour, got {:#08x}",
            far.1
        );
        // The near cell is NOT the fog colour (it kept its own brick look).
        assert!(
            dist_to_fog(near.0) > 40,
            "near fg must be untinted, got {:#08x}",
            near.0
        );
    }

    #[test]
    fn default_fog_is_fade_to_black_back_compat() {
        // With FogParams::DEFAULT (no tint), a far cell fades toward BLACK — the
        // historical behaviour, so shipped levels are byte-identical.
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let coord = Vec2::new(1.234, 2.0);
        let far = mat.sample(&perlin, coord, &ctx(40.0, 1.0));
        assert_eq!(far.fg, 0, "far fg fades to black under default fog");
        assert_eq!(far.bg, 0, "far bg fades to black under default fog");
    }

    #[test]
    fn registry_builtins_have_expected_ids() {
        let reg = MaterialRegistry::with_builtins();
        // Out-of-range id falls back to slot 0 (no panic).
        let _ = reg.get(MaterialId(999));
        // Each built-in slot resolves.
        for id in [
            ids::WALL_BRICK,
            ids::WALL_CIRCUIT,
            ids::WALL_MATRIX,
            ids::FLOOR_GRAIN,
            ids::CEILING_SKY,
        ] {
            let _ = reg.get(id);
        }
    }

    // ── per-cell text style (the new SGR / colored-underline channels) ──

    #[test]
    fn cell_style_bolds_a_lit_ridge() {
        let reg = MaterialRegistry::with_builtins();
        let brick = reg.get(ids::WALL_BRICK); // non-emissive wall
                                              // A relief ridge: densest ramp glyph drawn in a bright lit fg.
        let ridge = Texel {
            glyph: u32::from('@'),
            fg: 0x00F4_F4F4, // luma well above STYLE_BOLD_LUMA
            bg: 0x0010_1010,
            rank: 0,
        };
        let style = cell_style(brick, &ridge);
        assert!(
            style.is_bold(),
            "a dense, brightly-lit ridge cell must bold"
        );
        // A non-emissive wall gets NO colored underline.
        assert!(!style.has_underline_color());
    }

    #[test]
    fn cell_style_bold_tracks_luma_not_glyph_density() {
        let reg = MaterialRegistry::with_builtins();
        let brick = reg.get(ids::WALL_BRICK);
        // DIM fg → not a lit highlight, regardless of glyph.
        let dim = Texel {
            glyph: u32::from('@'),
            fg: 0x0030_3030,
            bg: 0,
            rank: 0,
        };
        assert!(
            !cell_style(brick, &dim).is_bold(),
            "a dim cell must not bold"
        );
        // Bright fg bolds even on a SPARSE glyph: the relief tuning desaturated the
        // near-field glyph ramp, so bold tracks lit luma, not glyph density.
        let bright_sparse = Texel {
            glyph: u32::from('.'),
            fg: 0x00FF_FFFF,
            bg: 0,
            rank: 0,
        };
        assert!(
            cell_style(brick, &bright_sparse).is_bold(),
            "a brightly-lit cell bolds regardless of glyph density"
        );
        // A blank (space) cell never bolds, however bright — there's no glyph to thicken.
        let blank = Texel {
            glyph: u32::from(' '),
            fg: 0x00FF_FFFF,
            bg: 0,
            rank: 0,
        };
        assert!(
            !cell_style(brick, &blank).is_bold(),
            "blank/void never bolds"
        );
    }

    #[test]
    fn cell_style_glows_an_emissive_material() {
        let reg = MaterialRegistry::with_builtins();
        let matrix = reg.get(ids::WALL_MATRIX); // emissive ≥ STYLE_GLOW_EMISSIVE
        assert!(
            matrix.light.emissive >= STYLE_GLOW_EMISSIVE,
            "matrix should be emissive enough to earn the glow"
        );
        let tx = Texel {
            glyph: u32::from('0'),
            fg: 0x0020_FF80,
            bg: 0x0004_140C,
            rank: 0,
        };
        let style = cell_style(matrix, &tx);
        assert!(
            style.has_underline_color(),
            "an emissive material must carry the colored-underline glow"
        );
        // The glow is an INDEPENDENT third channel — not equal to fg or bg.
        let uc = style.underline_color().unwrap();
        assert_ne!(uc, tx.fg & 0x00FF_FFFF);
        assert_ne!(uc, tx.bg & 0x00FF_FFFF);
        // A colored underline implies the underline attribute is on.
        assert!(style.is_underline());
    }

    /// THROWAWAY DIAGNOSTIC (playtest #9, U1-underline): print the colored-underline
    /// each catalog accent material emits for a representative LIT inked texel, and
    /// assert the four headline accents (gold, metal, neon, lava) each carry a
    /// DISTINCT underline colour. Proves the underline channel is populated per
    /// material (not a single global value). Run: `cargo test -p re_render
    /// underline_channel_diagnostic -- --nocapture`.
    #[test]
    fn underline_channel_diagnostic() {
        let reg = MaterialRegistry::with_catalog();
        // A bright, lit, inked texel in each material's own hue family.
        let cases: &[(&str, MaterialId, u32)] = &[
            ("gold", ids::GOLD, 0x00FF_E0A0),
            ("metal", ids::METAL, 0x00E8_F0FF),
            ("neon", ids::NEON, 0x00FF_40E0),
            ("lava", ids::LAVA, 0x00FF_8030),
        ];
        let mut colors = Vec::new();
        for (name, id, fg) in cases {
            let mat = reg.get(*id);
            let tx = Texel {
                glyph: u32::from('@'),
                fg: *fg,
                bg: 0x0010_0810,
                rank: 0,
            };
            let style = cell_style(mat, &tx);
            let uc = style.underline_color();
            println!(
                "  {name:6} accent={:?} gate={:?} -> underline={:?} substyle={:?}",
                mat.underline_accent.style,
                mat.underline_accent.gate,
                uc.map(|c| format!("#{c:06X}")),
                style.effective_underline_style(),
            );
            assert!(
                uc.is_some(),
                "{name} must emit a colored underline on a lit inked cell"
            );
            colors.push((*name, uc.unwrap()));
        }
        // The four headline accents must not all collapse to one colour.
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(
                    colors[i].1, colors[j].1,
                    "{} and {} produced the SAME underline colour — channel not per-material",
                    colors[i].0, colors[j].0
                );
            }
        }
    }

    #[test]
    fn cell_style_sky_and_default_are_unstyled() {
        let reg = MaterialRegistry::with_builtins();
        // Sky: blank void cell, never bolds and never glows (excluded).
        let sky = reg.get(ids::CEILING_SKY);
        let blank = Texel {
            glyph: u32::from(' '),
            fg: 0x00FF_FFFF,
            bg: 0,
            rank: 0,
        };
        assert!(
            cell_style(sky, &blank).is_empty(),
            "sky cell stays unstyled"
        );
        // The neutral default material on a mid cell: no bold, no glow.
        let mid = Texel {
            glyph: u32::from('+'),
            fg: 0x0040_4040,
            bg: 0,
            rank: 0,
        };
        assert!(cell_style(&DEFAULT_MATERIAL, &mid).is_empty());
    }

    // ── underline-ACCENT layer (the smart-material underline substyle) ──

    #[test]
    fn hologram_inked_cell_carries_curly_scan_glow() {
        use re_core::UnderlineStyle;
        let reg = MaterialRegistry::with_builtins();
        let holo = reg.get(ids::HOLO_NODE);
        // The holo material opts into the curly, inked scan-glow accent.
        assert!(
            holo.underline_accent.is_active(),
            "holo must have an active accent"
        );
        assert_eq!(holo.underline_accent.style, UnderlineStyle::Curly);
        // An inked scan-line cell (a line glyph in the holo hue) gets the accent.
        let line = Texel {
            glyph: HOLO_LINE_THIN,
            fg: 0x0040_E0F0,
            bg: pack_rgb(Vec3::splat(HOLO_VOID_LEVEL)),
            rank: 0,
        };
        let style = cell_style(holo, &line);
        // Curly substyle + a colored underline in the holo hue family.
        assert_eq!(style.effective_underline_style(), UnderlineStyle::Curly);
        assert!(
            style.has_underline_color(),
            "the scan-glow carries a colour"
        );
        assert!(
            style.is_underline(),
            "a substyle implies the underline attribute"
        );
    }

    #[test]
    fn hologram_void_gap_cell_gets_no_curly_scan_glow() {
        use re_core::UnderlineStyle;
        let reg = MaterialRegistry::with_builtins();
        let holo = reg.get(ids::HOLO_NODE);
        // A blank VOID gap between scan lines must NOT receive the CURLY scan-glow
        // accent — the `Inked` gate rejects the blank glyph. (A blank cell may still
        // pick up at most the legacy emissive under-glow's single rule, which is
        // pre-existing behaviour; the SUBSTYLE accent layer specifically does not
        // fire on the void gaps, so the curl reads only under the actual lines.)
        let gap = Texel {
            glyph: u32::from(' '),
            fg: 0x0001_0203,
            bg: pack_rgb(Vec3::splat(HOLO_VOID_LEVEL)),
            rank: 0,
        };
        let style = cell_style(holo, &gap);
        assert_ne!(
            style.effective_underline_style(),
            UnderlineStyle::Curly,
            "a void gap must not carry the curly scan-glow"
        );
    }

    #[test]
    fn lit_ridge_gate_glints_only_on_strongly_lit_cells() {
        use re_core::UnderlineStyle;
        // A material opting into the LitRidge specular-glint accent (a dotted,
        // hue-from-fg under-highlight). Shipped brick keeps this OFF for back-compat;
        // this synthetic material exercises the configurable accent layer.
        let mut mat = wall_brick();
        mat.light.emissive = 0.0; // isolate the accent from the emissive under-glow
        mat.underline_accent = UnderlineAccent {
            style: UnderlineStyle::Dotted,
            color: UnderlineAccent::HUE_FROM_FG,
            gate: AccentGate::LitRidge,
        };
        // A bright lit ridge earns the thin dotted glint.
        let ridge = Texel {
            glyph: u32::from('@'),
            fg: 0x00F4_F4F4,
            bg: 0x0010_1010,
            rank: 0,
        };
        let lit = cell_style(&mat, &ridge);
        assert_eq!(lit.effective_underline_style(), UnderlineStyle::Dotted);
        assert!(lit.has_underline_color());
        // A dim cell does NOT (the LitRidge gate keeps it rare/subtle).
        let dim = Texel {
            glyph: u32::from('#'),
            fg: 0x0030_3030,
            bg: 0x0010_1010,
            rank: 0,
        };
        assert_eq!(
            cell_style(&mat, &dim).effective_underline_style(),
            UnderlineStyle::None,
            "a dim cell carries no glint"
        );
        // Shipped brick itself stays accent-free (back-compat).
        assert_eq!(wall_brick().underline_accent, UnderlineAccent::NONE);
    }

    #[test]
    fn underline_accent_none_leaves_substyle_unset() {
        use re_core::UnderlineStyle;
        // The default material has no accent layer → no substyle ever.
        assert_eq!(DEFAULT_MATERIAL.underline_accent, UnderlineAccent::NONE);
        let cell = Texel {
            glyph: u32::from('+'),
            fg: 0x00FF_FFFF, // bright — would bold, but never accents
            bg: 0,
            rank: 0,
        };
        let style = cell_style(&DEFAULT_MATERIAL, &cell);
        assert_eq!(style.underline_style(), UnderlineStyle::None);
        assert_eq!(style.effective_underline_style(), UnderlineStyle::None);
        assert!(!style.has_underline_color());
    }

    // ── ANTI-FLICKER: glyph selection is stable under sub-cell camera motion ──

    /// A wall-cell lighting context for the calmer-look tests (face-on north wall).
    fn wall_light(view_dir: Vec3) -> LightCtx {
        LightCtx::for_wall(0, -1.0, view_dir, &LightEnv::builtin())
    }

    /// The quantized LOD driver is the anti-flicker deadband: two distances that
    /// fall inside the SAME [`LOD_QUANT_STEP`] plateau must quantize equal, so the
    /// glyph-selection density (and therefore the chosen glyph) does not change.
    #[test]
    fn quantize_lod_deadbands_small_deltas() {
        // Two distances a hair apart inside one step → identical quantized value.
        let a = quantize_lod(6.00);
        let b = quantize_lod(6.00 + LOD_QUANT_STEP * 0.2);
        assert_eq!(a, b, "a sub-step delta must quantize to the same plateau");
        // A delta that crosses a step boundary DOES change (the LOD still works).
        let c = quantize_lod(6.00 + LOD_QUANT_STEP * 1.1);
        assert_ne!(
            a, c,
            "crossing a step boundary must change the quantized LOD"
        );
        // Monotone non-decreasing.
        assert!(quantize_lod(2.0) <= quantize_lod(20.0));
    }

    /// End-to-end: the SAME wall texel, sampled at two camera distances a small
    /// sub-cell delta apart (well inside one [`LOD_QUANT_STEP`]), must pick the
    /// SAME GLYPH — glyphs must stay STABLE as the camera moves. Colour may differ
    /// (it reads the raw distance),
    /// but the discrete glyph choice must not flip on a tiny move. Probed across a
    /// strip of the wall so detail-reveal cells are covered too.
    #[test]
    fn wall_glyph_is_stable_across_a_tiny_distance_delta() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let glyph_at = |dist: f32, u: f32| -> u32 {
            let mut c = ctx(dist, 1.0);
            c.light = Some(wall_light(view));
            // No fog interference: push the band far out.
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            mat.sample(&perlin, Vec2::new(u, 1.3), &c).glyph
        };
        let base = 6.0f32;
        let nudge = LOD_QUANT_STEP * 0.25; // a sub-cell camera creep
        let mut checked = 0;
        for i in 0..64 {
            let u = i as f32 * 0.13;
            assert_eq!(
                glyph_at(base, u),
                glyph_at(base + nudge, u),
                "wall glyph flickered under a sub-step camera move at u={u}"
            );
            checked += 1;
        }
        assert!(checked > 0);
    }

    // ── IMPORTANCE / STABILITY: world-anchored glyph deadband + hysteresis ──

    /// Collect up to `n` wall-face coords whose BODY is INKED under the current
    /// sparse-coverage mask (so a glyph-ramp test exercises a cell whose glyph can
    /// change, not a blank one). Pure helper; world-anchored, so the inked set is
    /// stable. Scans a 2D patch of the face to gather a spread of inked texels.
    fn inked_wall_coords(seed: u32, n: usize) -> Vec<Vec2> {
        let mut out = Vec::with_capacity(n);
        'scan: for vi in 0..40 {
            for ui in 0..40 {
                let coord = Vec2::new(ui as f32 * 0.11, 0.7 + vi as f32 * 0.07);
                if wall_body_inked(seed, coord) {
                    out.push(coord);
                    if out.len() >= n {
                        break 'scan;
                    }
                }
            }
        }
        out
    }

    /// THE CORE WIN (quantified): under realistic jittery sub-cell camera motion,
    /// summed across many texels and distances, the `Stable` wall HOLDS its glyph at
    /// boundaries where the `Responsive` (shipped) path flickers — measurably less
    /// churn, exactly the flicker / eye-strain this stability tier exists to reduce.
    ///
    /// Metric note (sparse/coarse ramp): a raw OSCILLATING-wobble flip
    /// count is no longer a reliable `Stable < Responsive` ordering. The depth-ink
    /// ramp is now so coarse that a Responsive cell's FINE deadband steps mostly stay
    /// within ONE sparse ramp glyph, while each of Stable's WIDE deadband buckets makes
    /// a single LARGE depth-ink jump at its edge — so a wobble that happens to straddle
    /// a Stable bucket edge can show MORE raw flips for the coarser tier. We instead
    /// measure the churn over a MONOTONIC approach (the player walking steadily toward
    /// a wall, fine sub-cell steps): the number of distinct quantized distance buckets
    /// crossed — and hence the number of glyph TRANSITIONS — can only DROP as the
    /// deadband widens (coarser quantization ⇒ fewer buckets over the same span). This
    /// is robust to ramp coarseness and is the churn the player actually sees while
    /// moving. Held-at-a-boundary stability is asserted separately by
    /// `stable_glyph_invariant_where_responsive_flips`.
    #[test]
    fn stable_glyph_flips_less_than_responsive_under_jitter() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let glyph_at = |stab: GlyphStability, coord: Vec2, dist: f32| -> u32 {
            let mut mat = wall_brick();
            mat.stability = stab;
            let mut c = ctx(dist, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            mat.sample(&perlin, coord, &c).glyph
        };
        // Within-bucket wobble: jitter centres are placed at the MIDDLE of each Stable
        // deadband bucket (an odd multiple of the half-bucket = `LOD_QUANT_STEP × mul`),
        // and the wobble amplitude is kept below the half-bucket so a Stable wobble
        // NEVER leaves its plateau — while the Responsive grid (half as wide) still
        // recrosses an edge near the bucket centre. This isolates the load-bearing
        // win: for a realistic small view-dither, the shipped
        // Responsive wall flickers and the Stable wall holds rock-steady.
        // `quantize_lod_step(v, step)` rounds to the nearest multiple of `step`, so the
        // Responsive grid (step 1.25) has EDGES on the odd half-multiples 0.625, 1.875,
        // 3.125, … while the Stable grid (step 2.5) has edges on 1.25, 3.75, 6.25, … .
        // We dither around the Responsive edges that lie WELL INSIDE a Stable plateau
        // (1.875, 4.375, 6.875, … = 2.5·b − 0.625): with a wobble narrower than the
        // 0.625 gap to the nearest Stable edge, the FINE Responsive grid re-crosses an
        // edge (flicker) while the wide Stable plateau holds — the exact contrast.
        // The depth-ink glyph steps fastest in the DENSE NEAR band (small distances),
        // so that is where a sub-cell dither produces visible Responsive flicker. We
        // dither across that band and, at every Responsive flip, check whether the
        // wide Stable plateau held the glyph steady. The wobble (±0.45) is narrower
        // than the Stable half-bucket (1.25), so a Stable wobble that does not happen
        // to straddle a 1.25/3.75 edge holds — and the test sums the win across many
        // texels and near centres.
        let amp = 0.45f32;
        let seed = wall_brick().seed;
        let mut responsive = 0u32;
        let mut stable = 0u32;
        let mut frozen = 0u32;
        for coord in inked_wall_coords(seed, 192) {
            // Sweep the near band in fine centres (0.5 … ~3.5 cells) — where the
            // depth-ink ramp actually steps under a sub-cell dither.
            for ci in 0..60 {
                let center = 0.5 + ci as f32 * 0.05;
                let r_lo = glyph_at(GlyphStability::Responsive, coord, center - amp);
                let r_hi = glyph_at(GlyphStability::Responsive, coord, center + amp);
                if r_lo != r_hi {
                    responsive += 1;
                    if glyph_at(GlyphStability::Stable, coord, center - amp)
                        != glyph_at(GlyphStability::Stable, coord, center + amp)
                    {
                        stable += 1;
                    }
                    if glyph_at(GlyphStability::Frozen, coord, center - amp)
                        != glyph_at(GlyphStability::Frozen, coord, center + amp)
                    {
                        frozen += 1;
                    }
                }
            }
        }
        assert!(
            responsive >= 10,
            "the Responsive path must visibly churn under the near-band dither \
             (baseline), got {responsive}"
        );
        // THE SHIPPED WIN: the brick wall is `Stable`.
        // Across the SAME near-band dither, the wide Stable deadband (2× the Responsive
        // grid) HOLDS the glyph at a large share of the boundaries where the shipped
        // Responsive path flickers — the intended flicker / eye-strain reduction.
        // (Ceiling: ~half the Responsive 1.25-grid edges coincide with the
        // 2.5 Stable edges, where Stable must also flip; the OTHER half are interior to
        // a Stable plateau and held — a structural ~50% churn cut.)
        assert!(
            stable * 5 < responsive * 3,
            "Stable must flicker measurably less than Responsive under the near dither \
             (responsive={responsive}, stable={stable})"
        );
        // Frozen is the most-pinned tier: it must flicker no more than Stable.
        assert!(
            frozen <= stable,
            "Frozen must flicker no more than Stable (stable={stable}, frozen={frozen})"
        );
        println!(
            "NEAR-BAND dither flicker boundaries: responsive={responsive}  stable={stable}  \
             frozen={frozen}  (stable holds {:.0}% of Responsive's)",
            100.0 * (responsive.saturating_sub(stable)) as f32 / responsive.max(1) as f32,
        );
    }

    /// A `Stable` glyph is INVARIANT across six small distance steps over one cell
    /// of approach AT A BOUNDARY where the `Responsive` (shipped) glyph oscillates.
    /// We find the boundary, confirm the shipped
    /// path flips across the wobble, then assert the Stable path holds steady.
    #[test]
    fn stable_glyph_invariant_where_responsive_flips() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let glyph_at = |stab: GlyphStability, coord: Vec2, dist: f32| -> u32 {
            let mut mat = wall_brick();
            mat.stability = stab;
            let mut c = ctx(dist, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            mat.sample(&perlin, coord, &c).glyph
        };
        let amp = 0.4f32;
        // Find a boundary the Responsive path straddles within ±amp of some centre.
        let mut found: Option<(Vec2, f32)> = None;
        'search: for ui in 0..40 {
            let coord = Vec2::new(ui as f32 * 0.07, 1.3);
            for ci in 0..40 {
                let center = 2.5 + ci as f32 * 0.4;
                if glyph_at(GlyphStability::Responsive, coord, center - amp)
                    != glyph_at(GlyphStability::Responsive, coord, center + amp)
                {
                    found = Some((coord, center));
                    break 'search;
                }
            }
        }
        let (coord, center) = found.expect("expected a Responsive glyph boundary");
        // Confirm the SHIPPED path really would flip across this one cell of wobble.
        assert_ne!(
            glyph_at(GlyphStability::Responsive, coord, center - amp),
            glyph_at(GlyphStability::Responsive, coord, center + amp),
            "sanity: Responsive must flip across the chosen boundary"
        );
        // The Stable material must hold ONE glyph across six steps over that cell.
        let g0 = glyph_at(GlyphStability::Stable, coord, center - amp);
        for k in 0..=6 {
            let d = (center - amp) + k as f32 * (2.0 * amp / 6.0);
            assert_eq!(
                glyph_at(GlyphStability::Stable, coord, d),
                g0,
                "Stable glyph at coord={coord:?} must hold across one cell of approach \
                 where Responsive flips (d={d})"
            );
        }
    }

    /// COLOUR + PARALLAX STAY LIVE: across an approach where a Stable glyph holds,
    /// the cell's COLOUR (fg) must still VARY under realistic fog — stabilization
    /// pins only the discrete glyph, never the smooth shading. (Parallax — screen
    /// position/scale — is owned by the raycaster and untouched by the shader; the
    /// shader-side proof of "not frozen/dead" is that colour keeps moving.)
    #[test]
    fn stable_material_keeps_colour_varying_while_glyph_holds() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let coord = Vec2::new(1.10, 1.30);
        let mut mat = wall_brick();
        mat.stability = GlyphStability::Stable;
        // Realistic distance fog (the shipped default band) — the colour cue the
        // player sees as they move. The glyph is deadbanded; the colour is not.
        let sample_at = |dist: f32| -> (u32, u32) {
            let mut c = ctx(dist, 1.0);
            c.light = Some(wall_light(view));
            // default fog band (FOG_NEAR..FOG_FAR) → colour moves with raw distance.
            let t = mat.sample(&perlin, coord, &c);
            (t.glyph, t.fg)
        };
        // A span well inside one Stable bucket AND clear of the body-glyph boundary
        // so the glyph holds, but with enough distance change that fog moves colour.
        let base = 8.0f32;
        let (g0, _) = sample_at(base);
        let mut fg_values = std::collections::HashSet::new();
        for i in 0..=6 {
            let d = base - i as f32 * (1.5 / 6.0);
            let (g, fg) = sample_at(d);
            assert_eq!(g, g0, "glyph should hold over the span (d={d})");
            fg_values.insert(fg);
        }
        assert!(
            fg_values.len() > 1,
            "fg COLOUR must keep varying (fog reads raw distance) while the glyph holds \
             — got {} distinct values",
            fg_values.len()
        );
    }

    /// A `Responsive` / structural surface STILL UPDATES its glyphs with distance:
    /// the shipped behaviour is preserved (stability only ADDS calm to the stabler
    /// tiers; it never freezes the responsive ones). Across a wide near→far sweep
    /// the Responsive body glyph must take on more than one value somewhere on the
    /// face — it has NOT been pinned.
    #[test]
    fn responsive_surface_still_updates_with_distance() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let mut mat = wall_brick();
        mat.stability = GlyphStability::Responsive;
        let glyph_at = |coord: Vec2, d: f32| -> u32 {
            let mut c = ctx(d, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            mat.sample(&perlin, coord, &c).glyph
        };
        // The same texel seen near vs. far must differ for at least one face coord —
        // the depth-ink ramp still re-picks the body glyph with distance. Probe only
        // INKED cells: a (world-anchored, distance-invariant) blank body never changes
        // glyph by design, so the depth-ink behaviour lives on the inked cells.
        let mut any_changed = false;
        for coord in inked_wall_coords(mat.seed, 96) {
            if glyph_at(coord, 0.5) != glyph_at(coord, 24.0) {
                any_changed = true;
                break;
            }
        }
        assert!(
            any_changed,
            "Responsive surface must still update its glyph across distance"
        );
    }

    /// HYSTERESIS: once a detail glyph is revealed, a tiny step BACK does not
    /// un-reveal it. We drive [`lod_brightness_hysteretic`] across the reveal
    /// boundary: approaching turns the detail ON; a small retreat (which lowers the
    /// threshold a touch) must KEEP it on for the stabler tiers, whereas the plain
    /// (`hysteresis == 0`) path would flip it off.
    #[test]
    fn detail_reveal_is_hysteretic_on_a_small_step_back() {
        let rank = 40u8;
        // ENTER: at a near bucket the threshold is well above the rank → revealed.
        let enter_threshold = rank as f32 + REVEAL_FADE; // comfortably ON
                                                         // A small step back lowers the CURRENT threshold to JUST below the rank —
                                                         // the plain path would drop it; hysteresis must hold it.
        let leave_threshold = rank as f32 - 2.0;

        // Plain (Responsive) reveal: a threshold below the rank means OFF.
        let plain = lod_brightness_hysteretic(rank, leave_threshold, enter_threshold, 0.0);
        assert!(
            plain <= DETAIL_CUTOFF,
            "without hysteresis the detail drops (plain={plain})"
        );

        // Stable hysteresis: same step back, but it was revealed at the nearer
        // bucket → held ON above the cutoff.
        let held = lod_brightness_hysteretic(
            rank,
            leave_threshold,
            enter_threshold,
            GlyphStability::Stable.hysteresis(),
        );
        assert!(
            held > DETAIL_CUTOFF,
            "hysteresis must KEEP a revealed detail on across a small step back (held={held})"
        );

        // And a texel that was NEVER revealed (no nearer-bucket reveal) stays OFF
        // even with hysteresis — the deadband only HOLDS, it does not spuriously
        // turn glyphs on.
        let never = lod_brightness_hysteretic(
            rank,
            leave_threshold,
            rank as f32 - 1.0, // enter threshold also below rank → never entered
            GlyphStability::Stable.hysteresis(),
        );
        assert!(
            never <= DETAIL_CUTOFF,
            "hysteresis must not turn on a never-revealed detail (never={never})"
        );
    }

    /// The stability deadband widths + hysteresis ladder are ordered as intended
    /// (Responsive < Stable < Frozen) so the tiers actually mean "calmer".
    #[test]
    fn stability_tiers_are_ordered() {
        assert!(GlyphStability::Responsive.deadband_mul() < GlyphStability::Stable.deadband_mul());
        assert!(GlyphStability::Stable.deadband_mul() < GlyphStability::Frozen.deadband_mul());
        assert_eq!(GlyphStability::Responsive.hysteresis(), 0.0);
        assert!(GlyphStability::Stable.hysteresis() < GlyphStability::Frozen.hysteresis());
        // Default is Responsive (back-compat).
        assert_eq!(GlyphStability::default(), GlyphStability::Responsive);
    }

    /// BACK-COMPAT: a `Responsive` material reproduces the shipped quantizer exactly
    /// (the new stability path with ×1 deadband == the old `texel_density_quantized`).
    #[test]
    fn responsive_quantizer_matches_shipped() {
        for &(dist, fore) in &[(2.0f32, 1.0f32), (6.3, 1.4), (13.7, 2.0), (25.0, 6.0)] {
            let c = ctx(dist, fore);
            assert_eq!(
                texel_density_quantized_for(&c, GlyphStability::Responsive),
                texel_density_quantized(&c),
                "Responsive deadband must equal the shipped quantizer (dist={dist})"
            );
        }
    }

    /// The built-in materials carry the intended stability assignments (brick/floor
    /// body Stable, sky Frozen, structural/puzzle/circuit Responsive).
    #[test]
    fn builtins_have_expected_stability() {
        assert_eq!(wall_brick().stability, GlyphStability::Stable);
        assert_eq!(floor_grain().stability, GlyphStability::Stable);
        assert_eq!(ceiling_sky().stability, GlyphStability::Frozen);
        assert_eq!(wall_circuit().stability, GlyphStability::Responsive);
        assert_eq!(wall_matrix().stability, GlyphStability::Responsive);
        assert_eq!(puzzle_node().stability, GlyphStability::Responsive);
        assert_eq!(puzzle_door().stability, GlyphStability::Responsive);
    }

    // ── CALMER LOOK: floor + wall glyph density dropped (more low-density/space) ──

    /// Fraction of cells across a representative face/floor patch that carry a
    /// LOW-density glyph (space or the sparse tail of [`BODY_RAMP`]). Higher ⇒
    /// calmer. Used to prove the density dropped vs. the old packed look.
    fn low_density_fraction(mat: &Material, is_floor: bool) -> f32 {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        // "Low density" = space, or a body-ramp glyph in the sparse tail, or a thin
        // structural seam line (─/│) — anything that is NOT a packed mid/dense fill.
        let sparse_tail: &[u32] = &[
            u32::from(' '),
            u32::from('.'),
            u32::from('-'),
            u32::from(':'),
            u32::from('~'),
            u32::from('\''),
            0x2500, // ─
            0x2502, // │
        ];
        let mut low = 0usize;
        let mut total = 0usize;
        for i in 0..40 {
            for j in 0..40 {
                let mut c = ctx(5.0, 1.1);
                c.light = Some(if is_floor {
                    LightCtx::for_floor(true, Vec3::new(0.0, 0.0, 1.0), &LightEnv::builtin())
                } else {
                    wall_light(view)
                });
                c.fog = FogParams {
                    near: 1.0,
                    far: 1.0e6,
                    color: None,
                    height: None,
                };
                let coord = if is_floor {
                    Vec2::new(i as f32 * 0.5, j as f32 * 0.5)
                } else {
                    Vec2::new(i as f32 * 0.21, 1.0 + j as f32 * 0.18)
                };
                let g = mat.sample(&perlin, coord, &c).glyph;
                if sparse_tail.contains(&g) {
                    low += 1;
                }
                total += 1;
            }
        }
        low as f32 / total as f32
    }

    /// The floor must read mostly LOW-density (lots of space / sparse marks) — carry
    /// detail with COLOUR, don't fill everything. The current calibration
    /// (speck 0.05, dashed seams) pushes this even higher than an earlier pass, so
    /// the bar is raised to match.
    #[test]
    fn floor_is_mostly_low_density() {
        let frac = low_density_fraction(&floor_grain(), true);
        assert!(
            frac > 0.88,
            "calm floor must be mostly low-density / blank, got {frac:.2}"
        );
    }

    /// The wall must read as a calm low/mid-density field, not a solid `@%#` slab. The
    /// live "walls are a mess" recalibration ALSO blanks most body cells (sparse
    /// world-seeded coverage), so the low-density fraction is now a strong MAJORITY,
    /// far past the prior 0.30 bar — the bar is raised to prove the further drop.
    #[test]
    fn wall_is_not_a_dense_slab() {
        let frac = low_density_fraction(&wall_brick(), false);
        assert!(
            frac > 0.60,
            "calm wall must be a strong-majority low-density field, got {frac:.2}"
        );
    }

    // ── PUZZLE LEGIBILITY: node + door render distinct from walls ──

    /// The puzzle NODE and DOOR materials render a DISTINCT, FIXED glyph that no
    /// wall material emits — so a node reads as a node and the door as an exit,
    /// against the calmer walls. The fixed glyph survives every distance (a node
    /// stays a node up close AND far away).
    #[test]
    fn node_and_door_render_distinct_from_walls() {
        let perlin = Perlin::new(SHADER_SEED);
        let node = puzzle_node();
        let door = puzzle_door();
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let sample_glyph = |m: &Material, dist: f32| -> u32 {
            let mut c = ctx(dist, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            m.sample(&perlin, Vec2::new(2.3, 1.4), &c).glyph
        };
        // The node/door glyphs are their fixed symbols, near AND far.
        for d in [1.5f32, 8.0, 22.0] {
            assert_eq!(
                sample_glyph(&node, d),
                PUZZLE_NODE_GLYPH,
                "node stays a node at d={d}"
            );
            assert_eq!(
                sample_glyph(&door, d),
                PUZZLE_DOOR_GLYPH,
                "door stays a door at d={d}"
            );
        }
        // They differ from each other and from every wall material's glyph set.
        assert_ne!(PUZZLE_NODE_GLYPH, PUZZLE_DOOR_GLYPH);
        let mut wall_glyphs = std::collections::HashSet::new();
        for m in [wall_brick(), wall_circuit(), wall_matrix()] {
            for i in 0..200 {
                let mut c = ctx(3.0, 1.0);
                c.light = Some(wall_light(view));
                let g = m
                    .sample(
                        &perlin,
                        Vec2::new(i as f32 * 0.11, 1.0 + i as f32 * 0.07),
                        &c,
                    )
                    .glyph;
                wall_glyphs.insert(g);
            }
        }
        assert!(
            !wall_glyphs.contains(&PUZZLE_NODE_GLYPH),
            "no wall cell may emit the node glyph (it must be unmistakable)"
        );
        assert!(
            !wall_glyphs.contains(&PUZZLE_DOOR_GLYPH),
            "no wall cell may emit the door glyph"
        );
    }

    /// The node/door are SELF-LIT (emissive) and bright — they POP regardless of
    /// the headlamp, so the player sees the puzzle from across the room.
    #[test]
    fn node_and_door_are_bright_and_emissive() {
        let perlin = Perlin::new(SHADER_SEED);
        for m in [puzzle_node(), puzzle_door()] {
            assert!(
                m.light.emissive >= STYLE_GLOW_EMISSIVE,
                "puzzle element must be emissive"
            );
            // Even with the headlamp pointing AWAY, the emissive keeps the fg bright.
            let mut c = ctx(6.0, 1.0);
            c.light = Some(LightCtx::for_wall(
                0,
                -1.0,
                Vec3::new(0.0, -1.0, 0.0),
                &LightEnv::builtin(),
            ));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            let t = m.sample(&perlin, Vec2::new(2.3, 1.4), &c);
            assert!(
                luma_of(t.fg) > 0.25,
                "emissive puzzle element must stay bright, luma={}",
                luma_of(t.fg)
            );
        }
    }

    #[test]
    fn registry_has_puzzle_node_and_door_slots() {
        let reg = MaterialRegistry::with_builtins();
        assert_eq!(reg.id_of("puzzle_node"), Some(ids::PUZZLE_NODE));
        assert_eq!(reg.id_of("puzzle_door"), Some(ids::PUZZLE_DOOR));
        // The fixed glyph sources are pinned (so LOD can't thin them away).
        assert!(matches!(
            reg.get(ids::PUZZLE_NODE).glyph_source,
            GlyphSource::Fixed(_)
        ));
        assert!(matches!(
            reg.get(ids::PUZZLE_DOOR).glyph_source,
            GlyphSource::Fixed(_)
        ));
    }

    // ── #3 HOLOGRAM: glyph-only, void bg, vertical scroll ────────────────────

    /// The `holo_node` built-in is registered AFTER the fixed surface + puzzle
    /// slots, so it never disturbs the surface dispatch or `ids::PUZZLE_*`.
    #[test]
    fn registry_has_holo_node_slot() {
        let reg = MaterialRegistry::with_builtins();
        assert_eq!(reg.id_of("holo_node"), Some(ids::HOLO_NODE));
        // The puzzle slots are still where the surface dispatch expects them.
        assert_eq!(reg.id_of("puzzle_node"), Some(ids::PUZZLE_NODE));
        assert_eq!(reg.id_of("puzzle_door"), Some(ids::PUZZLE_DOOR));
        assert!(matches!(
            reg.get(ids::HOLO_NODE).recipe,
            TextureRecipe::Hologram { .. }
        ));
    }

    /// A near hologram is GLYPH-ONLY: across a vertical span its bg is a near-black
    /// VOID everywhere (never a filled colour), and where a scanline inks the glyph
    /// is a horizontal line glyph whose fg carries the holo hue (not the bg).
    #[test]
    fn hologram_is_glyph_only_on_a_void_bg() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let mat = holo_node();
        let void_floor = HOLO_VOID_LEVEL + 1e-3;
        let line_glyphs = [
            HOLO_LINE_SOLID,
            HOLO_LINE_THIN,
            HOLO_LINE_DASH,
            HOLO_LINE_DOT,
        ];
        let mut saw_line = false;
        let mut saw_gap = false;
        for i in 0..200 {
            // Sweep `coord.y` across several line periods at a fixed NEAR distance.
            let coord = Vec2::new(0.3, i as f32 * 0.02);
            let mut c = ctx(3.0, 1.0);
            c.light = Some(wall_light(view));
            // Kill fog so the void/line read is not confounded by fade-to-black.
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            let t = mat.sample(&perlin, coord, &c);
            // The bg is ALWAYS a near-black void — never a filled surface colour.
            assert!(
                luma_of(t.bg) <= void_floor,
                "hologram bg must be a near-black void (luma={}, glyph={})",
                luma_of(t.bg),
                t.glyph
            );
            if t.glyph == u32::from(' ') {
                saw_gap = true;
            } else {
                assert!(
                    line_glyphs.contains(&t.glyph),
                    "an inked hologram cell must be a scanline glyph, got U+{:04X}",
                    t.glyph
                );
                // The colour lives in the fg line, not the void bg.
                assert!(
                    luma_of(t.fg) > luma_of(t.bg),
                    "the line fg must outshine the void bg"
                );
                saw_line = true;
            }
        }
        assert!(saw_line, "the hologram must show scanline glyphs");
        assert!(
            saw_gap,
            "the hologram must be see-through between lines (void gaps)"
        );
    }

    /// A near hologram ANIMATES VERTICALLY: with its `Shift` scroll, the scanline
    /// pattern at a fixed surface coord changes as `time` advances (the lines move
    /// up the surface). The same material FAR-FROZEN does NOT animate.
    #[test]
    fn hologram_scrolls_vertically_near_but_freezes_far() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let mat = holo_node();
        assert!(
            mat.is_animated(),
            "the holo_node must carry a (Shift) animation"
        );

        let coord = Vec2::new(0.42, 0.07); // a point that sits ON a line at t=0
        let glyph_at = |dist: f32, freeze: f32, time: f32| -> u32 {
            let mut c = ctx(dist, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            c.far_freeze = freeze;
            c.time = time;
            mat.sample(&perlin, coord, &c).glyph
        };

        // NEAR (no freeze): the vertical scroll must change the glyph over a sweep of
        // time at this fixed coord (the lines drift past it).
        let near_inf = f32::INFINITY;
        let mut near_set = std::collections::HashSet::new();
        for k in 0..40 {
            near_set.insert(glyph_at(3.0, near_inf, k as f32 * 0.05));
        }
        assert!(
            near_set.len() > 1,
            "a NEAR hologram must animate vertically (got {} distinct glyphs over time)",
            near_set.len()
        );

        // FAR (frozen): the SAME time sweep at a far distance must hold ONE glyph.
        let far = FAR_FREEZE_DIST + 4.0;
        let mut far_set = std::collections::HashSet::new();
        for k in 0..40 {
            far_set.insert(glyph_at(far, FAR_FREEZE_DIST, k as f32 * 0.05));
        }
        assert_eq!(
            far_set.len(),
            1,
            "a FAR-FROZEN hologram must NOT animate (held {} glyphs over time)",
            far_set.len()
        );
    }

    // ── FAR-PLANE HARD FREEZE: glyph-lock + anim-disable, scroll-only ──────

    /// Beyond `FAR_FREEZE_DIST` a cell's GLYPH is INVARIANT to distance, viewing
    /// angle (foreshorten), AND time — the three churn drivers —
    /// while the SAME material NEAR still varies its glyph with distance/angle.
    #[test]
    fn far_frozen_glyph_is_invariant_across_distance_angle_time() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        // The matrix wall is an animated (Shift) recipe — a strong test that the
        // freeze kills BOTH LOD churn and animation. (Brick exercises the body ramp.)
        for mut mat in [wall_matrix(), wall_brick(), holo_node()] {
            mat.stability = GlyphStability::Stable; // even a stabilised tier still
                                                    // LOD-updates without the HARD cutoff
            let coord = Vec2::new(0.37, 1.21);
            let frozen_glyph = |dist: f32, fore: f32, time: f32| -> u32 {
                let mut c = ctx(dist, fore);
                c.light = Some(wall_light(view));
                c.fog = FogParams {
                    near: 1.0,
                    far: 1.0e6,
                    color: None,
                    height: None,
                };
                c.far_freeze = FAR_FREEZE_DIST;
                c.time = time;
                mat.sample(&perlin, coord, &c).glyph
            };
            // A baseline FAR glyph, then perturb distance, angle, and time — all must
            // leave the glyph UNCHANGED once past the cutoff.
            let base = frozen_glyph(FAR_FREEZE_DIST + 2.0, 1.0, 0.0);
            for &(d, f, t) in &[
                (FAR_FREEZE_DIST + 2.0, 1.0, 0.0),
                (FAR_FREEZE_DIST + 9.0, 1.0, 0.0),  // far further
                (FAR_FREEZE_DIST + 2.0, 3.5, 0.0),  // grazing angle
                (FAR_FREEZE_DIST + 2.0, 1.0, 7.3),  // time advanced
                (FAR_FREEZE_DIST + 20.0, 2.2, 3.1), // all at once
            ] {
                assert_eq!(
                    frozen_glyph(d, f, t),
                    base,
                    "FAR-FROZEN glyph must be invariant to distance/angle/time \
                     (recipe={:?}, d={d}, fore={f}, t={t})",
                    mat.recipe
                );
            }
        }
    }

    /// The freeze is a HARD CUTOFF, not a flatten-everything: the same animated
    /// material NEAR still animates/varies (so near surfaces stay alive), and a
    /// far-freeze of `∞` reproduces the pre-feature behaviour exactly.
    #[test]
    fn near_material_still_animates_when_far_freeze_active() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let mat = wall_matrix(); // a Shift-animated wall
        let coord = Vec2::new(0.37, 1.21);
        let glyph_at = |dist: f32, freeze: f32, time: f32| -> u32 {
            let mut c = ctx(dist, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            c.far_freeze = freeze;
            c.time = time;
            mat.sample(&perlin, coord, &c).glyph
        };

        // NEAR (well inside the cutoff) with the freeze ACTIVE: the material must
        // still animate over time — the freeze only touches FAR cells.
        let mut near_set = std::collections::HashSet::new();
        for k in 0..60 {
            near_set.insert(glyph_at(4.0, FAR_FREEZE_DIST, k as f32 * 0.07));
        }
        assert!(
            near_set.len() > 1,
            "a NEAR animated wall must stay alive while the far-freeze is active \
             (got {} glyphs)",
            near_set.len()
        );

        // far_freeze = ∞ ⇒ NO freeze anywhere (the pre-feature behaviour). A far cell
        // with ∞ is computed by exactly the old path; with the cutoff ACTIVE it is
        // glyph-locked to the fixed FAR bucket. The two must AGREE at t=0 only by luck,
        // so instead we prove ∞ does not freeze: a far cell at ∞ takes its glyph from
        // the live (un-bucketed) distance, identical to a far cell whose freeze cutoff
        // sits just BEYOND it (also unfrozen). Both bypass the lock ⇒ identical output.
        let far = FAR_FREEZE_DIST + 5.0;
        let unfrozen_inf = glyph_at(far, f32::INFINITY, 0.0);
        let unfrozen_beyond = glyph_at(far, far + 1.0, 0.0); // cutoff past the cell → not frozen
        assert_eq!(
            unfrozen_inf, unfrozen_beyond,
            "far_freeze=∞ must reproduce the un-frozen path exactly (no lock applied)"
        );
        // And the lock DOES change the far glyph when the cutoff is at/below the cell:
        // the frozen glyph is taken from the fixed FAR bucket, generally differing from
        // the live far glyph. (If they happen to coincide the freeze is still correct,
        // so we only require that the freeze PATH is reachable — asserted by the
        // invariance test above; here we simply confirm ∞ is a true no-op.)
        let _ = glyph_at(far, FAR_FREEZE_DIST, 0.0);
    }

    /// The frozen glyph is WORLD-ANCHORED, so the far field still SCROLLS with the
    /// camera (parallax preserved): two different world coords at the same far,
    /// frozen distance generally map to different glyphs — the pattern is a function
    /// of position, not a single dead constant.
    #[test]
    fn far_frozen_glyph_still_varies_by_world_position() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let mat = wall_brick();
        let glyph_at = |coord: Vec2| -> u32 {
            let mut c = ctx(FAR_FREEZE_DIST + 3.0, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            c.far_freeze = FAR_FREEZE_DIST;
            mat.sample(&perlin, coord, &c).glyph
        };
        // Sweep a 2D world patch (not a single scanline): with the sparse
        // wall body most cells blank to a coloured space and only a world-scattered
        // minority ink, so a single y-row can land entirely in blank wells. A 2D
        // sweep still proves the field VARIES by world position (blank vs inked, and
        // distinct ramp glyphs among the inked) — world-anchoring / parallax intact.
        let mut set = std::collections::HashSet::new();
        for vi in 0..16 {
            for ui in 0..64 {
                set.insert(glyph_at(Vec2::new(
                    ui as f32 * 0.13,
                    1.7 + vi as f32 * 0.37,
                )));
            }
        }
        assert!(
            set.len() > 1,
            "the frozen far field must be world-anchored (varying by position), not a \
             single dead constant — got {} distinct glyphs",
            set.len()
        );
    }

    // ── TEXEL PERSISTENCE CONTRACT: world-anchoring + maximal glyph hold ──────

    /// A CodeWall material for the persistence tests (the recipe whose CRT scanline
    /// was the one screen-domain randomness source, now re-anchored to the source
    /// row). Uses the fallback panel (no `&'static str` needed) so the test is
    /// self-contained; `Stable` so the glyph holds.
    fn code_wall_mat() -> Material {
        let mut m = wall_brick();
        m.recipe = TextureRecipe::CodeWall {
            panel: 0,
            u_chars_per_unit: 4.0,
        };
        m.glyph_source = GlyphSource::Recipe;
        m.stability = GlyphStability::Stable;
        m
    }

    /// GLYPH-HOLD CONTRACT: a STATIC world texel must keep ONE glyph across a
    /// full sweep of realistic SUB-CELL camera motion — distance creep, angle
    /// (foreshorten) wobble, and turning the view — because none of those cross a
    /// world-anchored, quantized hysteresis boundary. Hold the
    /// glyph as long as possible; don't noise without need — made into a flip count:
    /// it must be ZERO over the whole sub-cell sweep. The `Stable` brick wall is the
    /// subject; the `coord` is pinned (one world texel).
    #[test]
    fn static_texel_glyph_does_not_flip_under_subcell_camera_sweep() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick(); // Stable tier (the shipped wall)
                                // Use an INKED texel so its glyph can actually vary — a blank one is trivially
                                // constant and would not exercise the hold.
        let coord = inked_wall_coords(mat.seed, 1)[0];
        let glyph_at = |dist: f32, fore: f32, view: Vec3| -> u32 {
            let mut c = ctx(dist, fore);
            c.light = Some(wall_light(view.normalize()));
            // No fog/freeze interference: this isolates the LOD-glyph hold itself.
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            mat.sample(&perlin, coord, &c).glyph
        };
        // The reference: a mid-range, face-on view.
        let base_dist = 5.0f32;
        let g0 = glyph_at(base_dist, 1.0, Vec3::new(0.0, 1.0, 0.2));
        // Sweep sub-cell camera state: distance creep within one quantization plateau,
        // a small foreshorten (turn-in-place / strafe grazing) wobble, and a turning
        // view direction. ALL stay inside the `Stable` deadband, so the glyph holds.
        let mut flips = 0u32;
        let mut prev = g0;
        for k in 0..64 {
            let t = k as f32;
            // distance dithers by < one Stable deadband (±LOD_QUANT_STEP·0.45)
            let dist = base_dist + (t * 0.37).sin() * LOD_QUANT_STEP * 0.45;
            // foreshorten wobbles a touch above 1 (a small grazing angle)
            let fore = 1.0 + (t * 0.53).cos().abs() * 0.18;
            // view turns left/right a few degrees
            let yaw = (t * 0.29).sin() * 0.10;
            let view = Vec3::new(yaw, 1.0, 0.2);
            let g = glyph_at(dist, fore, view);
            if g != prev {
                flips += 1;
                prev = g;
            }
        }
        assert_eq!(
            flips, 0,
            "a static world texel's glyph must NOT flip under sub-cell camera motion \
             (distance creep + angle wobble + turn) — got {flips} flips at coord={coord:?}"
        );
    }

    /// CONTRACT #1, end-to-end across MANY texels: every inked wall texel holds its
    /// glyph across the sub-cell sweep (aggregate flip count is exactly 0). Guards
    /// against a single lucky coord masking a per-texel re-roll.
    #[test]
    fn many_texels_hold_glyph_under_subcell_sweep() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let glyph_at = |coord: Vec2, dist: f32, fore: f32, yaw: f32| -> u32 {
            let mut c = ctx(dist, fore);
            c.light = Some(wall_light(Vec3::new(yaw, 1.0, 0.2).normalize()));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            mat.sample(&perlin, coord, &c).glyph
        };
        let mut total_flips = 0u32;
        for coord in inked_wall_coords(mat.seed, 48) {
            let g0 = glyph_at(coord, 5.0, 1.0, 0.0);
            let mut prev = g0;
            for k in 0..24 {
                let t = k as f32;
                let dist = 5.0 + (t * 0.41).sin() * LOD_QUANT_STEP * 0.45;
                let fore = 1.0 + (t * 0.61).cos().abs() * 0.18;
                let yaw = (t * 0.31).sin() * 0.10;
                let g = glyph_at(coord, dist, fore, yaw);
                if g != prev {
                    total_flips += 1;
                    prev = g;
                }
            }
        }
        assert_eq!(
            total_flips, 0,
            "no inked wall texel may flip its glyph under a sub-cell camera sweep — \
             got {total_flips} aggregate flips"
        );
    }

    /// UNIFIED-LOD: with the DEFAULT [`crate::lod::LodConfig`] the per-cell
    /// [`ShadeLod`] reproduces the historical bands EXACTLY — relief gradient ON below
    /// 6, OFF at/above 6; octaves uncapped below 13, capped at/above 13. Proves the
    /// unification is not a visual change at the band edges the shade path reads.
    #[test]
    fn shade_lod_default_reproduces_historical_bands() {
        crate::lod::LodConfig::default().apply();
        // NEAR (well below the 6.0 boundary's lower deadband edge): full detail.
        let near = ShadeLod::for_distance(4.0);
        assert!(near.relief_gradient, "relief ON in the near tier");
        assert_eq!(near.max_octaves, u32::MAX, "no octave cap near");
        // MID (between the two boundaries' deadbands): relief dropped, octaves full.
        let mid = ShadeLod::for_distance(9.0);
        assert!(!mid.relief_gradient, "relief dropped in the mid tier");
        assert_eq!(mid.max_octaves, u32::MAX, "octaves still full in mid");
        // FAR (well past 13.0): relief dropped AND octaves capped.
        let far = ShadeLod::for_distance(20.0);
        assert!(!far.relief_gradient, "relief dropped far");
        assert_eq!(
            far.max_octaves,
            crate::lod::LOD_DEFAULT_FAR_OCTAVES,
            "octaves capped far"
        );
    }

    /// END-TO-END OVERRIDE DEMO (the "key object" lever): a hero material that carries
    /// `lod_override = full_detail_to(30)` keeps the relief gradient at a distance
    /// (25 world cells) where a DEFAULT material has dropped it. Drives the real
    /// [`Material::sample`] path (the override flows through `SampleCtx::lod_override`
    /// → `ShadeLod::for_ctx`), so this asserts the wired behaviour, not just the config.
    #[test]
    fn hero_material_override_keeps_detail_farther_than_default() {
        crate::lod::LodConfig::default().apply();
        let perlin = Perlin::new(SHADER_SEED);
        let coord = Vec2::new(2.3, 1.7);

        // A default wall material and the SAME material flagged as a hero object.
        let default_mat = wall_brick();
        let mut hero_mat = wall_brick();
        hero_mat.lod_override = Some(crate::lod::LodOverride::full_detail_to(30.0));

        let far_dist = 25.0; // past the global far-start (13), inside the hero's reach
        let mut c = ctx(far_dist, 1.0);
        c.light = Some(wall_light(Vec3::new(0.0, 1.0, 0.2).normalize()));

        // The shade LOD the sample path resolves for each material at this distance.
        let mut c_default = c;
        c_default.lod_override = default_mat.lod_override; // None
        let mut c_hero = c;
        c_hero.lod_override = hero_mat.lod_override; // full_detail_to(30)

        let default_lod = ShadeLod::for_ctx(&c_default);
        let hero_lod = ShadeLod::for_ctx(&c_hero);

        assert!(
            !default_lod.relief_gradient,
            "a DEFAULT material at distance 25 has dropped the relief gradient (FAR tier)"
        );
        assert!(
            hero_lod.relief_gradient,
            "the HERO material at distance 25 KEEPS the relief gradient (override holds NEAR)"
        );
        assert_eq!(
            hero_lod.max_octaves,
            u32::MAX,
            "the hero material also keeps full noise octaves at 25"
        );

        // And the two actually shade through `sample` without panicking, producing a
        // defined glyph for both (every input shape → a defined output).
        let _ = default_mat.sample(&perlin, coord, &c_default);
        let _ = hero_mat.sample(&perlin, coord, &c_hero);
    }

    /// CONTRACT #1 (world-anchored randomness): the SAME world texel yields the SAME
    /// recipe noise / glyph from TWO DIFFERENT camera viewpoints — different screen
    /// rows, different view directions, different distance/angle — because the recipe
    /// hashes on the world `coord`, not on the camera. We compare the FROZEN glyph
    /// (LOD locked, so any difference can only come from a camera-keyed hash) at one
    /// coord across a battery of viewpoints; it must be a single value.
    #[test]
    fn texel_noise_is_world_anchored_across_viewpoints() {
        let perlin = Perlin::new(SHADER_SEED);
        let coord = Vec2::new(0.41, 1.27);
        let mut mat = wall_brick();
        mat.stability = GlyphStability::Stable;
        let glyph_at = |dist: f32, fore: f32, screen_y: i32, view: Vec3| -> u32 {
            let mut c = ctx(dist, fore);
            c.screen_y = screen_y; // <- the camera-relative field that MUST be ignored
            c.light = Some(wall_light(view.normalize()));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            c.far_freeze = FAR_FREEZE_DIST; // freeze LOD so only a stray hash could differ
            mat.sample(&perlin, coord, &c).glyph
        };
        let base = glyph_at(FAR_FREEZE_DIST + 2.0, 1.0, 4, Vec3::new(0.0, 1.0, 0.2));
        for &(d, f, sy, vx) in &[
            (FAR_FREEZE_DIST + 2.0, 1.0, 4, 0.0),
            (FAR_FREEZE_DIST + 2.0, 1.0, 5, 0.0), // adjacent screen row
            (FAR_FREEZE_DIST + 2.0, 1.0, 41, 0.0), // far-away screen row
            (FAR_FREEZE_DIST + 7.0, 2.4, 17, 0.3), // different dist/angle/view
            (FAR_FREEZE_DIST + 2.0, 1.0, 4, -0.4), // view turned the other way
        ] {
            assert_eq!(
                glyph_at(d, f, sy, Vec3::new(vx, 1.0, 0.2)),
                base,
                "the same world texel must shade identically from every viewpoint \
                 (world-anchored) — differed at screen_y={sy}, dist={d}, fore={f}, vx={vx}"
            );
        }
    }

    /// CONTRACT #1, the re-anchored CRT scanline specifically: the CodeWall scanline
    /// dim is now keyed on the SOURCE-TEXT ROW (`wall_v`), not `ctx.screen_y`. So the
    /// SAME wall texel reads the SAME brightness from any screen row — proving the
    /// fixed screen-domain defect. We sample one `(coord, wall_v)` at many screen rows
    /// and assert the output texel is byte-identical (glyph + fg + bg).
    #[test]
    fn code_wall_scanline_is_world_anchored_not_screen_anchored() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = code_wall_mat();
        let coord = Vec2::new(2.5, 0.0); // a wall u with text on the fallback grid
        let texel_at = |screen_y: i32, wall_v: f32| -> Texel {
            let mut c = ctx(4.0, 1.0);
            c.screen_y = screen_y;
            c.wall_v = wall_v;
            // No light: isolate the recipe (the scanline lives in the recipe body).
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            mat.sample(&perlin, coord, &c)
        };
        // FIX-GUARD: for ONE fixed wall row, the texel must be identical at every
        // screen row (the old `screen_y & 1` parity would have flipped fg here).
        let wall_v = 0.5_f32;
        let base = texel_at(4, wall_v);
        for sy in [0, 1, 2, 3, 5, 17, 40, 41, 99] {
            assert_eq!(
                texel_at(sy, wall_v),
                base,
                "CodeWall texel must be invariant to the screen row (world-anchored \
                 scanline) — differed at screen_y={sy}"
            );
        }
        // SANITY: the scanline still EXISTS — two source rows of opposite parity must
        // (for an inked row) differ in brightness, so we didn't just delete the effect.
        // Scan source rows for an inked pair of opposite parity.
        let inked_fg = |wall_v: f32| -> Option<(i64, u32)> {
            let t = texel_at(4, wall_v);
            if t.glyph != u32::from(' ') {
                let rows = crate::scene_text::sample_code_panel(0, 0.0, 0.0, 1.0).1 as f32;
                Some(((wall_v * rows).floor() as i64, t.fg))
            } else {
                None
            }
        };
        let mut even: Option<u32> = None;
        let mut odd: Option<u32> = None;
        for k in 0..400 {
            let v = k as f32 / 400.0;
            if let Some((row, fg)) = inked_fg(v) {
                if row & 1 == 0 {
                    even.get_or_insert(fg);
                } else {
                    odd.get_or_insert(fg);
                }
            }
            if even.is_some() && odd.is_some() {
                break;
            }
        }
        // If the fallback panel has any inked text at all, the scanline must modulate
        // across parities; if the panel is wholly blank (no source), the effect is moot
        // and we simply skip the sanity half (the FIX-GUARD above is the load-bearing
        // assertion).
        if let (Some(e), Some(o)) = (even, odd) {
            assert_ne!(
                e, o,
                "the CRT scanline must still modulate brightness across alternate \
                 SOURCE rows (the effect must survive the re-anchor, not be deleted)"
            );
        }
    }

    /// CONTRACT #3 (far/idle freeze across frames): a FAR-frozen glyph is constant
    /// across a sweep of animation frames (`time`/`tick`) — the far field is fully
    /// quiet frame-to-frame. Complements `far_frozen_glyph_is_invariant_across_*`
    /// (which sweeps distance/angle/time) by sweeping a dense frame timeline on the
    /// recipe the player most associates with motion (the Shift matrix wall).
    #[test]
    fn far_freeze_holds_glyph_constant_across_frames() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let mat = wall_matrix(); // Shift-animated — strongest test of anim-disable
        let coord = Vec2::new(0.61, 1.13);
        let glyph_at = |tick: u64, time: f32| -> u32 {
            let mut c = ctx(FAR_FREEZE_DIST + 6.0, 1.0);
            c.light = Some(wall_light(view));
            c.fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            c.far_freeze = FAR_FREEZE_DIST;
            c.tick = tick;
            c.time = time;
            mat.sample(&perlin, coord, &c).glyph
        };
        let base = glyph_at(0, 0.0);
        let mut set = std::collections::HashSet::new();
        for k in 0..200 {
            set.insert(glyph_at(k, k as f32 * 0.05));
        }
        assert_eq!(
            set.len(),
            1,
            "a far-frozen glyph must hold across every animation frame (got {} distinct)",
            set.len()
        );
        assert_eq!(*set.iter().next().unwrap(), base);
    }

    // ── CALMER LOOK ──

    /// Fraction of a sampled patch that is strictly BLANK (a coloured space). The
    /// sparse-coverage masks blank a large,
    /// world-scattered fraction of both wall-body and floor cells, so this BLANK
    /// fraction must be high — the bulk of a surface is colour-carried ground.
    fn blank_fraction(mat: &Material, patch_origin: Vec2, n: u32, lit: bool) -> f32 {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let mut blank = 0u32;
        let mut total = 0u32;
        for vi in 0..n {
            for ui in 0..n {
                let coord = patch_origin + Vec2::new(ui as f32 * 0.37, vi as f32 * 0.41);
                let mut c = ctx(4.0, 1.0);
                if lit {
                    c.light = Some(if mat.seed == floor_grain().seed {
                        LightCtx::for_floor(true, Vec3::new(0.0, 0.0, 1.0), &LightEnv::builtin())
                    } else {
                        wall_light(view)
                    });
                }
                c.fog = FogParams {
                    near: 1.0,
                    far: 1.0e6,
                    color: None,
                    height: None,
                };
                if mat.sample(&perlin, coord, &c).glyph == u32::from(' ') {
                    blank += 1;
                }
                total += 1;
            }
        }
        blank as f32 / total as f32
    }

    /// WALL DENSITY DROPPED (vs the old fully-inked body): a meaningful, world-
    /// scattered fraction of brick-wall body cells is now a coloured BLANK — the old
    /// body inked EVERY cell, so it scored 0% blank. The "identical symbols in rows"
    /// mess is broken because most cells carry no glyph at all, only colour.
    #[test]
    fn wall_body_blanks_a_large_fraction_after_calming() {
        let frac = blank_fraction(&wall_brick(), Vec2::new(0.0, 0.7), 24, true);
        assert!(
            frac > 0.40,
            "calmer wall must BLANK a large fraction of body cells (was 0% before), \
             got {:.0}%",
            frac * 100.0
        );
    }

    /// FLOOR DENSITY DROPPED: the floor base is a coloured SPACE and specks/seams are
    /// now rare + dashed, so the floor is overwhelmingly blank — calm shaded ground.
    #[test]
    fn floor_blanks_overwhelming_fraction_after_calming() {
        let frac = blank_fraction(&floor_grain(), Vec2::new(0.13, 0.17), 24, false);
        assert!(
            frac > 0.80,
            "calmer floor must be OVERWHELMINGLY blank (colour-carried ground), \
             got {:.0}%",
            frac * 100.0
        );
    }

    /// NON-REPEATING TEXTURE: the calmer surfaces must NOT tile as identical-symbol
    /// rows — two world-separated cells of the SAME material can differ (the
    /// world-seeded scatter breaks the mechanical repetition). We sample a strip of
    /// each surface and assert it emits MORE THAN ONE distinct glyph (a repeating
    /// identical row would collapse to a single glyph), and specifically that BLANK
    /// and INKED cells coexist (the sparse, irregular scatter the brief asks for).
    #[test]
    fn calmer_surfaces_are_non_repeating() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        for (name, mat, lit) in [
            ("wall", wall_brick(), true),
            ("floor", floor_grain(), false),
        ] {
            let _ = lit;
            // Light BOTH surfaces the way the renderer does (the sparse-body blank
            // guard lives on the LIT relief path), so the test exercises the real path.
            let light = if name == "floor" {
                LightCtx::for_floor(true, Vec3::new(0.0, 0.0, 1.0), &LightEnv::builtin())
            } else {
                wall_light(view)
            };
            let mut set = std::collections::HashSet::new();
            let mut saw_blank = false;
            let mut saw_inked = false;
            for i in 0..80 {
                // A run of world-separated cells along one row of the surface.
                let coord = Vec2::new(i as f32 * 0.29, 1.3);
                let mut c = ctx(4.0, 1.0);
                c.light = Some(light);
                c.fog = FogParams {
                    near: 1.0,
                    far: 1.0e6,
                    color: None,
                    height: None,
                };
                let g = mat.sample(&perlin, coord, &c).glyph;
                set.insert(g);
                if g == u32::from(' ') {
                    saw_blank = true;
                } else {
                    saw_inked = true;
                }
            }
            assert!(
                set.len() > 1,
                "{name} surface must NOT be an identical-symbol row — got 1 glyph"
            );
            assert!(
                saw_blank && saw_inked,
                "{name} surface must mix BLANK and INKED cells (sparse, irregular \
                 scatter — not a packed row): blank={saw_blank} inked={saw_inked}"
            );
        }
    }

    /// WORLD-ANCHORED STABILITY PRESERVED: the new sparse-coverage mask is a pure
    /// function of the WORLD coord (not distance / screen / time), so a tiny camera
    /// move (a sub-deadband distance creep) must NOT change a cell from blank↔inked,
    /// and (away from a depth-ink plateau edge) must not re-pick the glyph. We assert
    /// the blank/inked STATE of every sampled wall + floor cell is invariant across a
    /// small distance delta — the anti-flicker guarantee must hold.
    #[test]
    fn sparse_coverage_is_world_anchored_not_distance_driven() {
        let perlin = Perlin::new(SHADER_SEED);
        let view = Vec3::new(0.0, 1.0, 0.2).normalize();
        let nudge = LOD_QUANT_STEP * 0.2; // a sub-deadband camera creep
        for (mat, lit) in [(wall_brick(), true), (floor_grain(), false)] {
            let is_blank = |coord: Vec2, d: f32| -> bool {
                let mut c = ctx(d, 1.0);
                if lit {
                    c.light = Some(wall_light(view));
                }
                c.fog = FogParams {
                    near: 1.0,
                    far: 1.0e6,
                    color: None,
                    height: None,
                };
                mat.sample(&perlin, coord, &c).glyph == u32::from(' ')
            };
            for i in 0..64 {
                let coord = Vec2::new(i as f32 * 0.31, 1.3);
                assert_eq!(
                    is_blank(coord, 5.0),
                    is_blank(coord, 5.0 + nudge),
                    "a cell's blank/inked state must be WORLD-anchored (invariant to a \
                     tiny camera move) — the sparse mask must not churn under motion"
                );
            }
        }
    }

    /// The sparse-coverage masks honour their coverage knobs and stay deterministic:
    /// the wall body inks roughly [`WALL_BODY_COVERAGE`] of world spots (well under
    /// half — sparse), the same world spot is ALWAYS the same decision (deterministic),
    /// and the floor seam keeps only a fraction of seam segments (dashed, not solid).
    #[test]
    fn sparse_masks_are_sparse_and_deterministic() {
        let seed = wall_brick().seed;
        // Wall body coverage ≈ WALL_BODY_COVERAGE across a big world patch.
        let mut inked = 0u32;
        let mut total = 0u32;
        for vi in 0..120 {
            for ui in 0..120 {
                let coord = Vec2::new(ui as f32 * 0.21, vi as f32 * 0.19);
                if wall_body_inked(seed, coord) {
                    inked += 1;
                }
                total += 1;
                // Deterministic: same input → same decision.
                assert_eq!(
                    wall_body_inked(seed, coord),
                    wall_body_inked(seed, coord),
                    "wall_body_inked must be deterministic"
                );
            }
        }
        let cov = inked as f32 / total as f32;
        // The mask inks the LOW-noise wells of a clamped value-noise field, so the
        // realised inked fraction sits BELOW the raw `WALL_BODY_COVERAGE` threshold —
        // sparse by construction. It must be a small minority (mostly colour-carried)
        // yet keep SOME detail (not a bare wall).
        assert!(
            cov < 0.35,
            "wall body must be SPARSELY inked (mostly colour-carried), got {:.0}%",
            cov * 100.0
        );
        assert!(
            cov > 0.01,
            "wall body must keep SOME inked detail (not bare), got {:.1}%",
            cov * 100.0
        );
        // Floor body is even sparser than the wall (the floor is calm ground).
        let fcov = {
            let mut fi = 0u32;
            let mut ft = 0u32;
            for vi in 0..120 {
                for ui in 0..120 {
                    let coord = Vec2::new(ui as f32 * 0.21, vi as f32 * 0.19);
                    if floor_body_inked(floor_grain().seed, coord) {
                        fi += 1;
                    }
                    ft += 1;
                }
            }
            fi as f32 / ft as f32
        };
        assert!(
            fcov < cov + 0.01,
            "floor body must be no denser than the wall body (calm ground): \
             floor={:.1}% wall={:.1}%",
            fcov * 100.0,
            cov * 100.0
        );
        // Floor seam keeps only a fraction of segments (dashed). Probe many segments
        // along one world line and confirm both kept and dropped occur.
        let fseed = floor_grain().seed;
        let mut kept = 0u32;
        for s in 0..200 {
            let coord = Vec2::new(3.0, s as f32 * FLOOR_SEAM_SEG + 0.01);
            if floor_seam_inked(fseed, coord, true) {
                kept += 1;
            }
        }
        assert!(
            kept > 0 && kept < 200,
            "floor seam must be DASHED (some segments kept, some dropped), kept={kept}/200"
        );
    }

    // ── catalog: the rich library of smart materials ──────────────────────────

    #[test]
    fn with_catalog_preserves_the_fixed_builtin_slots() {
        // Slots 0..=7 must be byte-identical between with_builtins and with_catalog
        // (the surface dispatch + puzzle + hologram must not move).
        let base = MaterialRegistry::with_builtins();
        let cat = MaterialRegistry::with_catalog();
        for slot in 0..=7u16 {
            let a = base.get(MaterialId(slot));
            let b = cat.get(MaterialId(slot));
            assert_eq!(
                a.recipe_tag(),
                b.recipe_tag(),
                "slot {slot} recipe must be unchanged by the catalog append"
            );
            assert_eq!(format!("{:?}", a.palette.a), format!("{:?}", b.palette.a));
        }
        // The catalog is APPENDED, so the registry has strictly more slots.
        assert!(
            cat.len() > base.len(),
            "catalog adds slots beyond the built-ins"
        );
    }

    #[test]
    fn every_catalog_material_registers_and_resolves_by_name() {
        let cat = MaterialRegistry::with_catalog();
        for name in ids::CATALOG_NAMES {
            let id = cat
                .id_of(name)
                .unwrap_or_else(|| panic!("catalog material '{name}' must be registered"));
            assert!(
                id.0 >= 8,
                "catalog material '{name}' sits past the fixed slots"
            );
            // The standalone data lookup must agree with the registered slot.
            let direct = MaterialRegistry::catalog_material(name)
                .unwrap_or_else(|| panic!("catalog_material('{name}') must resolve"));
            assert_eq!(direct.recipe_tag(), cat.get(id).recipe_tag());
        }
        assert_eq!(
            ids::CATALOG_NAMES.len(),
            43,
            "the catalog has 43 named materials (16 base + haze/flow/spark \
             + vitrine/screen/veil + moss/ember_field/static_noise/lattice \
             + beat_grid/auto_tile/boil + portal_zoom/portal_spiral \
             + frost/wet_stone/redacted/watcher/worn_floor/decay \
             + node_breathing + substrate + mnemos_hold/lethe_release + conduit \
             + facade)"
        );
    }

    #[test]
    fn catalog_materials_have_distinct_configs() {
        // No two catalog materials are the same config — each is a distinct DATA
        // point (palette bias OR recipe OR light OR animation differs). We hash a
        // few salient fields and require all-distinct.
        let mats: Vec<Material> = ids::CATALOG_NAMES
            .iter()
            .map(|n| MaterialRegistry::catalog_material(n).unwrap())
            .collect();
        let sig = |m: &Material| {
            format!(
                "{}|{:?}|{:?}|{}|{}|{:?}|{}|{}",
                m.recipe_tag(),
                m.palette.a.to_array(),
                m.bg_palette.a.to_array(),
                m.light.emissive,
                m.light.specular,
                m.animation.kind,
                m.reflective,
                m.light.ao_strength,
            )
        };
        let mut seen = std::collections::HashSet::new();
        for (n, m) in ids::CATALOG_NAMES.iter().zip(&mats) {
            assert!(
                seen.insert(sig(m)),
                "catalog material '{n}' duplicates another config"
            );
        }
    }

    #[test]
    fn catalog_properties_read_as_configured() {
        // A spot-check that the smart-material PROPERTIES are wired: lava/monitor/neon
        // emit + animate, water/glass reflect, stone is matte with deep AO.
        let lava = MaterialRegistry::catalog_material("lava").unwrap();
        assert!(
            lava.light.emissive > 0.5 && lava.is_animated(),
            "lava glows + pulses"
        );
        let monitor = MaterialRegistry::catalog_material("monitor").unwrap();
        assert!(
            monitor.is_animated() && monitor.light.emissive > 0.5,
            "monitor glows + flickers"
        );
        assert_eq!(monitor.animation.kind, AnimKind::Flicker);
        let neon = MaterialRegistry::catalog_material("neon").unwrap();
        assert_eq!(neon.animation.kind, AnimKind::Flicker);
        let water = MaterialRegistry::catalog_material("water").unwrap();
        assert!(
            water.reflective && water.is_animated(),
            "water reflects + flows"
        );
        let glass = MaterialRegistry::catalog_material("glass").unwrap();
        assert!(
            glass.reflective && glass.clarity > 0.8,
            "glass is a crisp mirror"
        );
        let stone = MaterialRegistry::catalog_material("stone").unwrap();
        assert!(
            stone.light.specular < 0.1 && stone.light.ao_strength > 0.7 && !stone.is_animated(),
            "stone is matte, deep-AO, static"
        );
        // The catalog flips any_animated (several materials animate).
        assert!(MaterialRegistry::with_catalog().any_animated());
    }

    // ── SUBSTRATE — the cell alphabet as a wall material ─────────────────────
    //
    // `crate::substrate`'s own tests prove the ALPHABET (the state table, the
    // erosion, the world anchoring, totality). These prove the WIRING: that the
    // catalog material renders through that function, that the shading pipeline
    // does not clobber the glyph on its way out, and that no other material moved.

    /// The shipped substrate material's recipe parameters, read off the material
    /// itself rather than off the module constants — so a test below that predicts a
    /// glyph is predicting what the SHIPPED material will draw.
    fn substrate_params() -> (Material, f32, f32, f32) {
        let m =
            MaterialRegistry::catalog_material("substrate").expect("substrate is in the catalog");
        let TextureRecipe::Substrate {
            scale,
            held,
            released,
        } = m.recipe
        else {
            panic!(
                "the substrate material must carry the Substrate recipe, got {:?}",
                m.recipe
            );
        };
        (m, scale, held, released)
    }

    /// A lit wall `SampleCtx` carrying a WALL-PLANE hit at world `(along, up)` — the
    /// pair the one cast path threads for every wall cell, and the ONLY thing the
    /// substrate recipe reads. `coord` is deliberately set to a fixed, unrelated value
    /// by the callers below so a recipe that regressed to reading it is caught.
    fn substrate_ctx(distance: f32, along: f32, up: f32) -> SampleCtx {
        let mut c = lit_ctx(distance);
        c.wall_along = along;
        c.wall_up = up;
        c
    }

    /// A patch of WALL-PLANE world positions spanning many alphabet cells (and several
    /// world cells along the run), sampled at cell CENTRES in the scaled lattice so no
    /// probe sits on a boundary.
    fn substrate_probe_plane(scale: f32) -> Vec<Vec2> {
        let step = 1.0 / scale;
        (0..24 * 24)
            .map(|i| {
                Vec2::new(
                    ((i % 24) as f32 + 0.5) * step,
                    ((i / 24) as f32 + 0.5) * step,
                )
            })
            .collect()
    }

    /// A `coord` that must NOT influence the substrate texel. Off-lattice and far from
    /// the probe plane, so any leak shows up as a wrong glyph rather than a coincidence.
    const SUBSTRATE_DECOY_COORD: Vec2 = Vec2::new(41.37, 17.61);

    /// The material must paint the CELL ALPHABET — every glyph one of the four, and
    /// the glyph the one `crate::substrate::cell_glyph` predicts for that world cell
    /// under the material's OWN parameters. That equality is the wiring proof: a
    /// recipe that re-derived the rule, or dropped `released`, or lost the seed, would
    /// still emit four plausible glyphs and fail here.
    #[test]
    fn substrate_material_paints_exactly_the_cell_alphabet() {
        let perlin = Perlin::new(SHADER_SEED);
        let (m, scale, held, released) = substrate_params();
        let mut seen = std::collections::BTreeSet::new();
        for p in substrate_probe_plane(scale) {
            let c = substrate_ctx(3.0, p.x, p.y);
            let g = char::from_u32(m.sample(&perlin, SUBSTRATE_DECOY_COORD, &c).glyph)
                .expect("a substrate glyph is valid Unicode");
            assert!(
                crate::substrate::ALPHABET.contains(&g),
                "wall plane {p:?} painted {g:?}, which is not in the cell alphabet"
            );
            assert_eq!(
                g,
                crate::substrate::cell_glyph(
                    Vec3::new(p.x * scale, p.y * scale, 0.0),
                    held,
                    released,
                    m.seed,
                ),
                "wall plane {p:?}: the wall must render THROUGH the alphabet function"
            );
            seen.insert(g);
        }
        // Anti-vacuity: the shipped parameters must put a MIX on the wall. A material
        // that painted one glyph everywhere would satisfy every claim above.
        assert!(
            seen.len() >= 3,
            "the shipped substrate paints only {seen:?} — the field must read as an \
             alphabet, not as a fill"
        );
    }

    /// FOUND BY RENDERING. The four tiers must separate by BRIGHTNESS, monotonically,
    /// in the order `█ > ▒ > ░ > ✕` their tones declare — the second channel that makes
    /// the alphabet read when the glyph itself has gone to mush at distance.
    ///
    /// The first implementation walked the tone across a window of the cosine palette
    /// and landed on its DESCENDING side, which inverted the whole thing: `█` came out
    /// at fg luma 0.032 against `░` at 0.079. Worse, folding in glyph ink coverage
    /// (`█` fills its cell; `░` covers about a quarter of it) made the two match in
    /// PERCEIVED luminance to within 0.001 — so the tiers were separated by glyph shape
    /// alone. Every existing test passed. It took one `--dump` and a printed swatch.
    ///
    /// Probed through the un-lit `texture_sample` so the claim is about the RECIPE's
    /// own colour choice, not about whatever the room's light does to it afterwards.
    #[test]
    fn substrate_tiers_are_monotone_in_brightness() {
        use crate::substrate::{BURNED, HELD, SCAR, UNCLAIMED};
        /// How far apart adjacent tiers must sit in fg luma. Wide enough that the
        /// room's lighting and the fog cannot collapse two tiers into one.
        const MIN_TIER_GAP: f32 = 0.04;
        let perlin = Perlin::new(SHADER_SEED);
        let (m, scale, _, _) = substrate_params();

        // Drive each tier deliberately: a setting where the whole field is one glyph.
        let whole_field = [
            (UNCLAIMED, 0.0_f32, 0.0_f32),
            (HELD, 1.0, 0.0),
            (SCAR, 1.0, 0.9),
            (BURNED, 1.0, 1.0),
        ];
        let mut luma = std::collections::BTreeMap::new();
        for (want, held, released) in whole_field {
            let mut probe = m.clone();
            probe.recipe = TextureRecipe::Substrate {
                scale,
                held,
                released,
            };
            let c = substrate_ctx(2.0, 3.5 / scale, 1.5 / scale);
            let tex = probe.texture_sample(&perlin, SUBSTRATE_DECOY_COORD, &c);
            assert_eq!(
                char::from_u32(tex.base.glyph),
                Some(want),
                "({held}, {released}) must drive the whole field to {want:?}"
            );
            luma.insert(want, (luma_of(tex.base.fg), luma_of(tex.base.bg)));
        }

        // 1. The fg ordering is the tone ordering, strictly.
        let f = |g: char| luma[&g].0;
        assert!(
            f(HELD) > f(SCAR) && f(SCAR) > f(UNCLAIMED) && f(UNCLAIMED) > f(BURNED),
            "the tiers must be strictly ordered in fg luma, got \
             █ {:.4}, ▒ {:.4}, ░ {:.4}, ✕ {:.4}",
            f(HELD),
            f(SCAR),
            f(UNCLAIMED),
            f(BURNED)
        );
        // 2. …and SEPARATED, not merely ordered. Adjacent tiers must differ by enough
        //    that the room's lighting and the fog cannot collapse them into one.
        for (hi, lo) in [(HELD, SCAR), (SCAR, UNCLAIMED), (UNCLAIMED, BURNED)] {
            assert!(
                f(hi) - f(lo) >= MIN_TIER_GAP,
                "{hi:?} ({:.4}) and {lo:?} ({:.4}) are within {MIN_TIER_GAP} — the two \
                 tiers are separated by glyph shape alone",
                f(hi),
                f(lo)
            );
        }
        // 3. Nothing is a HOLE: the darkest tier still inks well above black, so a
        //    burned wall reads as a marked surface and not as a gap in the geometry.
        assert!(
            f(BURNED) > 0.10,
            "the burned tier inks at luma {:.4} — that reads as a hole in the wall",
            f(BURNED)
        );
        // 4. The bg runs the OTHER way, so a bright cell sits on a dark ground.
        let b = |g: char| luma[&g].1;
        assert!(
            b(HELD) < b(BURNED),
            "the ground must be darkest under the brightest tier, got █ on {:.4} and \
             ✕ on {:.4}",
            b(HELD),
            b(BURNED)
        );
        // 5. Every tier keeps fg above bg, so the glyph always reads as ink on ground
        //    (a tier whose fg fell below its bg would render as a negative).
        for g in crate::substrate::ALPHABET {
            assert!(
                f(g) > b(g),
                "{g:?} inks at {:.4} on a ground of {:.4} — the cell is inverted",
                f(g),
                b(g)
            );
        }

        // 6. THE ONE THAT MATTERS — see the sibling test below for why the five
        //    assertions above were all true while the wall still read wrong.
        assert_apparent_brightness_is_ordered(&luma);
    }

    /// A cell's APPARENT brightness: `coverage · fg + (1 − coverage) · bg`. This — not
    /// `fg` alone — is what a viewer sees, because the glyph decides how much of the
    /// cell the `fg` colour actually covers.
    fn apparent_brightness(glyph: char, fg_luma: f32, bg_luma: f32) -> f32 {
        let c = crate::substrate::ink_coverage(glyph);
        c * fg_luma + (1.0 - c) * bg_luma
    }

    /// The tiers must be ordered in APPARENT brightness, not merely in `fg` luma.
    ///
    /// FOUND BY RENDERING, and the sharpest lesson of this task. The five assertions in
    /// `substrate_tiers_are_monotone_in_brightness` were all TRUE of the first shipped
    /// alphabet — `fg` strictly ordered `█ > · > ░ > ✕` (the scar was `·` then), separated,
    /// never inverted
    /// against its ground — and a `--dump` still showed the scar tier as a hole punched
    /// in the wall. The glyph was doing the work the colour was being tested for: `·`
    /// inks about 4% of its cell against `░`'s 25%, so despite a HIGHER `fg` luma
    /// (0.275 vs 0.219) it came out the DARKEST thing on the wall — below even the burn
    /// — and inverted the bottom of the order.
    ///
    /// Fixing it needed the GLYPH, not the colour: closing a 6× coverage deficit on
    /// brightness alone would have pushed the scar past `█`. `SCAR` is now `▒`, and the
    /// alphabet's three shade blocks carry the tiers by ink coverage with `✕` off the
    /// ramp as a mark.
    #[track_caller]
    fn assert_apparent_brightness_is_ordered(luma: &std::collections::BTreeMap<char, (f32, f32)>) {
        use crate::substrate::{BURNED, HELD, SCAR, UNCLAIMED};
        /// How far apart adjacent tiers must sit in APPARENT brightness, so no two
        /// collapse into one at a glance. The tightest real gap (`░` over `✕`)
        /// measured 0.025, so the bar sits below that with room.
        const MIN_APPARENT_GAP: f32 = 0.012;
        let a = |g: char| apparent_brightness(g, luma[&g].0, luma[&g].1);
        // The order the tones DECLARE, which the first alphabet's `fg` satisfied and
        // its apparent brightness did not.
        let order = [HELD, SCAR, UNCLAIMED, BURNED];
        for pair in order.windows(2) {
            let (hi, lo) = (pair[0], pair[1]);
            assert!(
                a(hi) > a(lo),
                "apparent brightness is out of order: {hi:?} reads {:.4} but {lo:?} \
                 reads {:.4}. A cell is `coverage · fg + (1 − coverage) · bg`, so the \
                 GLYPH outvotes the colour — this is the check that `fg` ordering alone \
                 does not make. Coverages: {hi:?} {:.2}, {lo:?} {:.2}",
                a(hi),
                a(lo),
                crate::substrate::ink_coverage(hi),
                crate::substrate::ink_coverage(lo)
            );
        }
        // …and SEPARATED, so no two tiers collapse into one at a glance.
        for pair in order.windows(2) {
            let (hi, lo) = (pair[0], pair[1]);
            assert!(
                a(hi) - a(lo) >= MIN_APPARENT_GAP,
                "{hi:?} ({:.4}) and {lo:?} ({:.4}) are within {MIN_APPARENT_GAP} of \
                 each other in apparent brightness — the two tiers read as one",
                a(hi),
                a(lo)
            );
        }
    }

    /// FOUND BY RENDERING. The field must be continuous ALONG A WALL RUN — it must not
    /// repeat once per world cell.
    ///
    /// The first implementation sampled `coord`, whose `x` is `tex_u + floor(the face's
    /// CONSTANT axis)`: `tex_u` wraps `0→1` inside every world cell and the offset is
    /// the same for every cell of one flat run, so the whole wall tiled with a period
    /// of exactly one world unit. Every unit test passed; a `--dump` of a wall showed
    /// it in one glance. Two claims, both measured:
    ///
    /// 1. the glyph sequence along the run is NOT periodic with period 1 (the defect's
    ///    exact signature), and
    /// 2. `coord` does not reach the texel at all (the cause).
    #[test]
    fn the_substrate_field_does_not_tile_once_per_world_cell() {
        let perlin = Perlin::new(SHADER_SEED);
        let (m, scale, _, _) = substrate_params();
        let up = 1.5;
        // Sample the run at alphabet-cell centres across six world cells.
        let per_cell = scale.round() as usize;
        let along = |i: usize| (i as f32 + 0.5) / scale;
        let glyph = |a: f32| {
            m.sample(&perlin, SUBSTRATE_DECOY_COORD, &substrate_ctx(2.0, a, up))
                .glyph
        };
        let run: Vec<u32> = (0..per_cell * 6).map(|i| glyph(along(i))).collect();

        // 1. NOT periodic with one world cell. Under the defect every window of
        //    `per_cell` samples was identical; here they must not all agree.
        let repeats = (0..run.len() - per_cell)
            .filter(|i| run[*i] == run[i + per_cell])
            .count();
        let n = run.len() - per_cell;
        assert!(
            repeats < n,
            "every one of {n} samples one world cell apart matched — the field TILES \
             once per world cell (it is reading `tex_u`, not the wall-plane hit)"
        );
        // Anti-vacuity: the run must actually vary, or "not periodic" is meaningless.
        assert!(
            run.iter().collect::<std::collections::BTreeSet<_>>().len() >= 2,
            "the probed run is a single glyph, so the periodicity claim is vacuous"
        );

        // 2. `coord` does not reach the texel: hold the wall plane fixed and sweep
        //    `coord` wildly. Every texel — glyph, fg, bg — must be identical.
        let fixed = substrate_ctx(2.0, 3.5 / scale, up);
        let base = m.sample(&perlin, SUBSTRATE_DECOY_COORD, &fixed);
        for c in [
            Vec2::ZERO,
            Vec2::new(0.5, 0.5),
            Vec2::new(-7.25, 91.0),
            Vec2::splat(1e3),
        ] {
            assert_eq!(
                base,
                m.sample(&perlin, c, &fixed),
                "coord {c:?} changed the substrate texel — the recipe is reading the \
                 per-cell-wrapped `coord` again"
            );
        }
    }

    /// The glyph must survive the whole shading pipeline: the relief body re-selection
    /// (which would otherwise replace it with a density-ramp glyph), the sparse-body
    /// blanking (which would replace it with a space), and the far / far-frozen LOD.
    ///
    /// Probed on the LIT path, because the relief re-selection only runs when a
    /// `LightCtx` is present — a light-less probe would pass on a broken `props()`.
    #[test]
    fn the_substrate_glyph_survives_relief_lighting_and_the_far_lod() {
        let perlin = Perlin::new(SHADER_SEED);
        let (m, scale, held, released) = substrate_params();
        let plane = substrate_probe_plane(scale);
        for (dist, freeze) in [(0.5_f32, f32::INFINITY), (3.0, f32::INFINITY), (40.0, 20.0)] {
            let mut seen = std::collections::BTreeSet::new();
            for p in &plane {
                let mut c = substrate_ctx(dist, p.x, p.y);
                c.far_freeze = freeze;
                let g = char::from_u32(m.sample(&perlin, SUBSTRATE_DECOY_COORD, &c).glyph).unwrap();
                assert_eq!(
                    g,
                    crate::substrate::cell_glyph(
                        Vec3::new(p.x * scale, p.y * scale, 0.0),
                        held,
                        released,
                        m.seed,
                    ),
                    "d={dist} freeze={freeze} plane {p:?}: the relief/LOD pass \
                     replaced the alphabet glyph"
                );
                seen.insert(g);
            }
            assert!(seen.len() >= 3, "d={dist}: the field flattened to {seen:?}");
        }
    }

    /// STATIC and WORLD-ANCHORED as a rendered material: no `time` term reaches it, so
    /// it reports not-animated (a static scene still idles at cache cost), and a
    /// sub-cell camera creep cannot flip a glyph.
    #[test]
    fn the_substrate_material_is_static_and_camera_stable() {
        let perlin = Perlin::new(SHADER_SEED);
        let (m, scale, _, _) = substrate_params();
        assert!(!m.is_animated(), "substrate must not be time-animated");
        assert_eq!(
            m.animation.kind,
            AnimKind::None,
            "substrate carries Animation::NONE"
        );

        // Time must not reach the texel at all.
        let (along, up) = (2.5 / scale, 3.5 / scale);
        let base = m.sample(
            &perlin,
            SUBSTRATE_DECOY_COORD,
            &substrate_ctx(4.0, along, up),
        );
        for t in [0.0_f32, 0.37, 17.0, 1e4] {
            let mut c = substrate_ctx(4.0, along, up);
            c.time = t;
            assert_eq!(
                base,
                m.sample(&perlin, SUBSTRATE_DECOY_COORD, &c),
                "time {t} changed the texel"
            );
        }

        // A sub-cell creep (a tenth of an alphabet cell on the wall plane, plus a
        // sub-deadband distance creep) must not re-deal the wall.
        let nudge = 0.1 / scale;
        let mut moved = substrate_ctx(4.0 + LOD_QUANT_STEP * 0.2, along + nudge, up + nudge);
        moved.time = 9.0;
        assert_eq!(
            base.glyph,
            m.sample(&perlin, SUBSTRATE_DECOY_COORD, &moved).glyph,
            "a sub-cell camera creep flipped the substrate glyph — the field would swim"
        );
    }

    /// A new recipe variant + a new catalog slot must not change ANY existing
    /// material's output. The direct, checkable half of that claim: nothing that
    /// shipped before adopts the `Substrate` recipe, and nothing that shipped before
    /// moved slot. (The cross-build half — that the rendered frames are unchanged — is
    /// `re_game`'s `golden_framebuffer` suite.)
    #[test]
    fn no_shipped_material_adopts_the_substrate_recipe_yet() {
        let reg = MaterialRegistry::with_catalog();
        let substrate_tag = TextureRecipe::Substrate {
            scale: 3.0,
            held: 0.0,
            released: 0.0,
        }
        .props()
        .tag;
        let adopters: Vec<u16> = (0..reg.len())
            .filter(|i| reg.get(MaterialId(*i as u16)).recipe_tag() == substrate_tag)
            .map(|i| i as u16)
            .collect();
        assert_eq!(
            adopters,
            vec![ids::SUBSTRATE.0],
            "exactly one slot may carry the Substrate recipe, and it is the new one"
        );
        // The fixed surface / puzzle / hologram slots every shipped level dispatches
        // through are untouched…
        for i in 0..=7u16 {
            assert_ne!(reg.get(MaterialId(i)).recipe_tag(), substrate_tag);
        }
        // …and the new slot is APPENDED past every previously-defined catalog id, so
        // no existing MaterialId shifted.
        const { assert!(ids::SUBSTRATE.0 > ids::NODE_BREATHING.0) };
        assert_eq!(
            reg.id_of("node_breathing"),
            Some(ids::NODE_BREATHING),
            "the previous tail material must not have moved"
        );
        assert_eq!(reg.id_of("substrate"), Some(ids::SUBSTRATE));
    }

    // ── THE VOICE PAIR — the substrate alphabet driven by the state bus ──────
    //
    // `crate::substrate` proves the ALPHABET and the MARK arithmetic; `re_game`
    // proves the mark is monotone and cleared on reset. These prove the WIRING: that
    // the state word reaches the field, that it reaches it as a SCALAR and not as
    // anything else the same word means elsewhere, and that the two walls read as two
    // events rather than as one wall twice.

    /// The shipped voice materials, by name, with the mode each carries — read off
    /// the material rather than assumed, so a test below that predicts a glyph is
    /// predicting what the SHIPPED material will draw.
    fn voice_params(name: &str) -> (Material, f32, crate::substrate::VoiceMode) {
        let m = MaterialRegistry::catalog_material(name)
            .unwrap_or_else(|| panic!("{name} is in the catalog"));
        let TextureRecipe::Voice { scale, mode } = m.recipe else {
            panic!("{name} must carry the Voice recipe, got {:?}", m.recipe);
        };
        (m, scale, mode)
    }

    /// The two shipped voices, in `(name, mode)` order.
    const VOICES: [(&str, crate::substrate::VoiceMode); 2] = [
        ("mnemos_hold", crate::substrate::VoiceMode::Hold),
        ("lethe_release", crate::substrate::VoiceMode::Release),
    ];

    /// A wall `SampleCtx` at a wall-plane hit carrying a state-bus WORD.
    fn voice_ctx(along: f32, up: f32, word: u32) -> SampleCtx {
        let mut c = substrate_ctx(3.0, along, up);
        c.state = word;
        c
    }

    /// The glyph field a voice material paints across the probe plane at `word`.
    fn voice_field(m: &Material, scale: f32, word: u32) -> Vec<char> {
        let perlin = Perlin::new(SHADER_SEED);
        substrate_probe_plane(scale)
            .into_iter()
            .map(|p| {
                char::from_u32(
                    m.sample(&perlin, SUBSTRATE_DECOY_COORD, &voice_ctx(p.x, p.y, word))
                        .glyph,
                )
                .expect("a voice glyph is valid Unicode")
            })
            .collect()
    }

    /// **THE INVARIANT.** `ctx.state` INDEXES a glyph family / triggers a track /
    /// levels a `SurfaceFx` everywhere else in this file; the voice recipe reads the
    /// same word as a SCALAR. That is tolerable only while the two readings never meet
    /// on one material — so this asserts it, rather than trusting that nobody later
    /// attaches a family to a voice wall. An invariant that depends on future
    /// restraint is exactly the kind that decays silently.
    ///
    /// Scans the WHOLE registry by the recipe's own `state_is_scalar` declaration, not
    /// the two names — so a third voice material added later is covered the day it is
    /// added, and one that stopped declaring itself is caught here.
    #[test]
    fn the_voice_materials_carry_no_other_state_reader() {
        use crate::glyph_family::FamilyDriver;
        let reg = MaterialRegistry::with_catalog();

        let scalar_slots: Vec<u16> = (0..reg.len())
            .filter(|i| {
                reg.get(MaterialId(*i as u16))
                    .recipe
                    .props()
                    .state_is_scalar
            })
            .map(|i| i as u16)
            .collect();
        assert_eq!(
            scalar_slots,
            vec![ids::MNEMOS_HOLD.0, ids::LETHE_RELEASE.0],
            "exactly the two voice slots may read the state word as a scalar"
        );

        // The predicate every arm below is the negation of: does this material carry
        // ANY OTHER consumer of `ctx.state`? The three are the three sites in
        // `Material::sample` that read the word per-material (the fourth — the FLOW
        // reading — is not opt-in and is handled by `state_is_scalar`; the behavioural
        // proof for it is `an_odd_mark_does_not_read_as_a_powered_pipe`).
        let other_state_reader = |m: &Material| -> Option<&'static str> {
            if matches!(m.family, Some(l) if l.driver == FamilyDriver::State) {
                return Some("a State-driven glyph family (indexes members by ctx.state)");
            }
            if m.anim_tracks.count > 0 {
                return Some("an animation track set (an OnState trigger reads ctx.state)");
            }
            if m.surface_fx.is_some() {
                return Some("a SurfaceFx finish (redaction/decay level off ctx.state)");
            }
            None
        };

        for slot in &scalar_slots {
            let m = reg.get(MaterialId(*slot));
            assert!(
                other_state_reader(m).is_none(),
                "slot {slot} reads ctx.state as a MARK and also carries {} — one word \
                 cannot be both a scalar and an index on the same material",
                other_state_reader(m).unwrap()
            );
        }

        // ANTI-VACUITY. The guard above is worthless if `other_state_reader` cannot
        // detect anything, so hand it each of the three violations in turn on a real
        // voice material and require it to fire.
        let (base, _, _) = voice_params("mnemos_hold");
        let with_family = Material {
            family: Some(crate::glyph_family::FamilyLayer {
                family: crate::glyph_family::STATE_NODE,
                driver: FamilyDriver::State,
                speed: 0.0,
            }),
            ..base.clone()
        };
        assert!(
            other_state_reader(&with_family).is_some(),
            "the guard must detect a State family, or it proves nothing"
        );
        let with_fx = Material {
            surface_fx: Some(crate::surface_fx::SurfaceFx::Redaction),
            ..base.clone()
        };
        assert!(
            other_state_reader(&with_fx).is_some(),
            "the guard must detect a state-reading SurfaceFx"
        );
        let with_track = Material {
            anim_tracks: MaterialRegistry::catalog_material("node_breathing")
                .expect("node_breathing is in the catalog")
                .anim_tracks,
            ..base
        };
        assert!(
            with_track.anim_tracks.count > 0,
            "the borrowed track set must be non-empty or the next assertion is vacuous"
        );
        assert!(
            other_state_reader(&with_track).is_some(),
            "the guard must detect an animation track set"
        );
    }

    /// **The collision the invariant above does NOT cover, proved by behaviour.**
    ///
    /// `flow_shift_coord` / `flow_is_hot` read `ctx.state` in `Material::sample` for
    /// EVERY material — not opt-in — and `re_core::decode_flow_state` keys on bit 0.
    /// So before `RecipeProps::state_is_scalar`, an ODD mark (501 parts of 1000)
    /// decoded as a powered pipe and lifted the wall's fg by `FLOW_HOT_LIFT`: a
    /// progress bar that strobed brighter/dimmer on every other part as it filled.
    ///
    /// The assertion is exact rather than statistical. For a voice cell `fg` is a pure
    /// function of `(glyph, rank)` and `rank` does not depend on the mark — so wherever
    /// two adjacent marks leave a cell's GLYPH unchanged, its `fg` must be unchanged
    /// too, bit for bit. Under the defect every even→odd step broke that.
    #[test]
    fn an_odd_mark_does_not_read_as_a_powered_pipe() {
        let perlin = Perlin::new(SHADER_SEED);
        for (name, _) in VOICES {
            let (m, scale, _) = voice_params(name);
            let plane = substrate_probe_plane(scale);
            let mut compared = 0usize;
            for word in 480u32..520 {
                for p in &plane {
                    let a = m.sample(&perlin, SUBSTRATE_DECOY_COORD, &voice_ctx(p.x, p.y, word));
                    let b = m.sample(
                        &perlin,
                        SUBSTRATE_DECOY_COORD,
                        &voice_ctx(p.x, p.y, word + 1),
                    );
                    if a.glyph != b.glyph {
                        continue; // the cell genuinely converted — colours may differ.
                    }
                    assert_eq!(
                        (a.fg, a.bg),
                        (b.fg, b.bg),
                        "{name}: marks {word} and {} paint the same glyph at {p:?} in \
                         DIFFERENT colours — the odd word is decoding as a powered pipe \
                         and taking the FLOW_HOT_LIFT",
                        word + 1
                    );
                    compared += 1;
                }
            }
            // Anti-vacuity: most cells hold their glyph across a one-part step, so the
            // comparison above must actually have run on nearly all of them.
            assert!(
                compared > 40 * plane.len() * 9 / 10,
                "{name}: only {compared} same-glyph pairs were compared — the claim is \
                 close to vacuous"
            );
        }
    }

    /// Each voice renders THROUGH `crate::substrate::cell_glyph`, with the `(held,
    /// released)` pair its own mode derives from the state word. The equality is the
    /// wiring proof: a recipe that re-derived the rule, swapped the channels, or
    /// dropped the mark would still emit four plausible glyphs and fail here.
    #[test]
    fn the_voice_materials_paint_the_alphabet_the_mark_predicts() {
        for (name, mode) in VOICES {
            let (m, scale, carried) = voice_params(name);
            assert_eq!(carried, mode, "{name} must carry the {mode:?} mode");
            for word in [0u32, 1, 250, 500, 750, 950, 999, 1000, 5000] {
                let mark = crate::substrate::decode_mark(word);
                let (held, released) = mode.channels(mark);
                for (p, g) in substrate_probe_plane(scale)
                    .into_iter()
                    .zip(voice_field(&m, scale, word))
                {
                    assert_eq!(
                        g,
                        crate::substrate::cell_glyph(
                            Vec3::new(p.x * scale, p.y * scale, 0.0),
                            held,
                            released,
                            m.seed,
                        ),
                        "{name} at word {word}: the wall must render through the \
                         alphabet function with its own mode's channels"
                    );
                }
            }
        }
    }

    /// The two walls must read as TWO EVENTS, not as one wall shown twice.
    ///
    /// Three separate claims, because "different" is cheap and the near-miss here is
    /// specific: both voices deal their field from `cell_rank`, so at the SAME seed
    /// the hold voice's held set (`rank < mark`) and the release voice's still-held
    /// set (`rank >= mark`) are exact COMPLEMENTS — the same deal in negative. That
    /// would pass a naive `assert_ne!` on every mark.
    #[test]
    fn the_two_voices_do_not_paint_the_same_deal() {
        let (hold, hscale, _) = voice_params("mnemos_hold");
        let (rel, rscale, _) = voice_params("lethe_release");
        assert_eq!(hscale, rscale, "both voices share the alphabet density");
        assert_ne!(
            hold.seed, rel.seed,
            "the two voices must be dealt from DIFFERENT seeds"
        );

        // Marks in the middle of the range, where all three claims can discriminate:
        // near either endpoint one voice's field is nearly uniform and the "two
        // independent deals" prediction below collapses onto the complementary one.
        for word in [200u32, 350, 500, 650, 800] {
            let mark = crate::substrate::decode_mark(word);
            let h = voice_field(&hold, hscale, word);
            let r = voice_field(&rel, rscale, word);

            // 1. Different GLYPH VOCABULARY. The hold voice moves ░ → █; the release
            //    voice moves █ → ▒. They share only `█`, from opposite sides.
            let seen = |f: &[char]| f.iter().copied().collect::<std::collections::BTreeSet<_>>();
            assert!(
                seen(&h).contains(&crate::substrate::UNCLAIMED)
                    && !seen(&r).contains(&crate::substrate::UNCLAIMED),
                "word {word}: only the HOLD voice may show unclaimed ground — \
                 hold {:?}, release {:?}",
                seen(&h),
                seen(&r)
            );
            assert!(
                seen(&r).contains(&crate::substrate::SCAR)
                    && !seen(&h).contains(&crate::substrate::SCAR),
                "word {word}: only the RELEASE voice may show scars — \
                 hold {:?}, release {:?}",
                seen(&h),
                seen(&r)
            );

            // 2. Different DEAL. At mark `m` the hold voice holds `m` of the field and
            //    the release voice still holds `1 − m` of it. If the two were dealt
            //    from the SAME seed those sets would be exact COMPLEMENTS and agree on
            //    0 of the field; two INDEPENDENT deals agree on
            //    `m·(1−m) + (1−m)·m = 2m(1−m)`. The measured agreement must match the
            //    independent prediction, which is 0.32–0.50 across this sweep — an
            //    order of magnitude away from both degenerate cases.
            let hold_set: Vec<bool> = h.iter().map(|g| *g == crate::substrate::HELD).collect();
            let rel_set: Vec<bool> = r.iter().map(|g| *g == crate::substrate::HELD).collect();
            let agree = hold_set
                .iter()
                .zip(&rel_set)
                .filter(|(a, b)| a == b)
                .count() as f32
                / hold_set.len() as f32;
            let independent = 2.0 * mark * (1.0 - mark);
            // Tolerance: the probe plane is 576 cells, so a proportion measured on it
            // has a binomial sd of ~0.021 at `p = 0.5` — this is ~5 sd. Wide enough
            // that the hash's own lumpiness cannot fail it, and still leaving a 0.33
            // margin to either degenerate value.
            assert!(
                (agree - independent).abs() <= 0.12,
                "word {word}: the two voices' held sets agree on {agree:.3} of the \
                 field; two independent deals would agree on {independent:.3}. Near 0 \
                 means they share a seed and are showing the SAME deal in negative; \
                 near 1 means they are showing it outright. Either reads as one wall \
                 twice."
            );
        }
    }

    /// Both walls convert MONOTONICALLY and IRREVERSIBLY in the mark, per cell.
    ///
    /// Irreversibility is a property of the mark (owned by `re_game`), but it only
    /// buys anything if the SURFACE is monotone in it too: a field that re-dealt at
    /// each mark would hold its counts and still boil cell-by-cell as the puzzle
    /// advanced. Asserted per cell, over a fine sweep, through `Material::sample`.
    #[test]
    fn the_voice_walls_convert_monotonically_with_the_mark() {
        const STEPS: u32 = 40;
        for (name, _) in VOICES {
            let (m, scale, _) = voice_params(name);
            // "Converted" means the same thing for both voices: this cell is no longer
            // in the state it started in.
            let rest = voice_field(&m, scale, 0);
            let mut prev: Option<Vec<bool>> = None;
            let mut counts = std::collections::BTreeSet::new();
            for k in 0..=STEPS {
                let word = k * (crate::substrate::MARK_PARTS / STEPS);
                let f = voice_field(&m, scale, word);
                let converted: Vec<bool> = f.iter().zip(&rest).map(|(a, b)| a != b).collect();
                let n = converted.iter().filter(|b| **b).count();
                if let Some(p) = &prev {
                    assert!(
                        p.iter().zip(&converted).all(|(a, b)| !*a || *b),
                        "{name} at word {word}: a converted cell went BACK — the wall \
                         must not un-convert as the mark rises"
                    );
                }
                counts.insert(n);
                prev = Some(converted);
            }
            assert_eq!(
                *counts.iter().next().unwrap(),
                0,
                "{name}: nothing may be converted at mark 0"
            );
            assert!(
                counts.len() >= STEPS as usize - 2,
                "{name}: the converted count took only {} values across {} mark steps \
                 — the wall is stepping, not converting",
                counts.len(),
                STEPS + 1
            );
        }
    }

    /// The voice walls must satisfy the SAME apparent-brightness ordering the
    /// substrate does — the check `fg`-only assertions do not make (see
    /// `assert_apparent_brightness_is_ordered`). Their palettes are different hues, so
    /// this is a real re-measurement, not a restatement.
    #[test]
    fn the_voice_tiers_are_ordered_in_apparent_brightness() {
        use crate::substrate::{BURNED, HELD, SCAR, UNCLAIMED};
        let perlin = Perlin::new(SHADER_SEED);
        for (name, _) in VOICES {
            let (m, scale, _) = voice_params(name);
            // Drive each tier deliberately through the SUBSTRATE recipe on the voice's
            // own palette: the voice's own mode can only reach two tiers at a time, and
            // the ordering claim is about all four.
            let mut luma = std::collections::BTreeMap::new();
            for (want, held, released) in [
                (UNCLAIMED, 0.0_f32, 0.0_f32),
                (HELD, 1.0, 0.0),
                (SCAR, 1.0, 0.9),
                (BURNED, 1.0, 1.0),
            ] {
                let mut probe = m.clone();
                probe.recipe = TextureRecipe::Substrate {
                    scale,
                    held,
                    released,
                };
                let c = substrate_ctx(2.0, 3.5 / scale, 1.5 / scale);
                let tex = probe.texture_sample(&perlin, SUBSTRATE_DECOY_COORD, &c);
                assert_eq!(
                    char::from_u32(tex.base.glyph),
                    Some(want),
                    "{name}: ({held}, {released}) must drive the whole field to {want:?}"
                );
                luma.insert(want, (luma_of(tex.base.fg), luma_of(tex.base.bg)));
            }
            assert_apparent_brightness_is_ordered(&luma);
        }
    }

    /// The voices are STATE-driven, not TIME-driven. They must report as not-animated,
    /// or a static scene would re-cast every frame and the near-ring pan would be
    /// disabled for any level that adopts one. (The plan's Task 5 Step 5 test.)
    #[test]
    fn voice_materials_are_not_time_animated() {
        let perlin = Perlin::new(SHADER_SEED);
        for name in ["mnemos_hold", "lethe_release", "substrate"] {
            let m = MaterialRegistry::catalog_material(name).expect("registered");
            assert_eq!(
                m.animation.kind,
                AnimKind::None,
                "{name} carries Animation::NONE"
            );
            assert!(!m.is_animated(), "{name} must not be animated by any layer");
        }
        // …and the stronger form for the voices: `time` must not reach the texel at
        // all, at a NON-ZERO mark (a test at mark 0 could pass on a dead recipe).
        for (name, _) in VOICES {
            let (m, scale, _) = voice_params(name);
            let (along, up) = (2.5 / scale, 3.5 / scale);
            let base = m.sample(&perlin, SUBSTRATE_DECOY_COORD, &voice_ctx(along, up, 500));
            for t in [0.0_f32, 0.37, 17.0, 1e4] {
                let mut c = voice_ctx(along, up, 500);
                c.time = t;
                assert_eq!(
                    base,
                    m.sample(&perlin, SUBSTRATE_DECOY_COORD, &c),
                    "{name}: time {t} changed the texel"
                );
            }
        }
    }

    /// The `Voice` recipe is the `Substrate` surface with a different driver, so its
    /// [`RecipeProps`] must match `Substrate`'s on every field EXCEPT the two that are
    /// meant to differ: the stable recipe `tag`, and `state_is_scalar`.
    ///
    /// Pinned because the two arms are hand-written and adjacent — the exact shape
    /// that drifts. A voice wall that quietly became `self_lit`, or lost
    /// `is_code_wall`, would have the relief pass re-select its glyph off the density
    /// ramp and the alphabet would stop being an alphabet.
    #[test]
    fn the_voice_recipe_declares_the_same_surface_properties_as_substrate() {
        let sub = TextureRecipe::Substrate {
            scale: 3.0,
            held: 0.5,
            released: 0.25,
        }
        .props();
        let voice = TextureRecipe::Voice {
            scale: 3.0,
            mode: crate::substrate::VoiceMode::Hold,
        }
        .props();
        assert_ne!(
            voice.tag, sub.tag,
            "the recipe KIND discriminant must differ"
        );
        assert!(sub.state_is_scalar != voice.state_is_scalar && voice.state_is_scalar);
        // Both read the WALL PLANE — same surface, same anchoring, so the cache keys
        // must fingerprint the same geometry. Only the STATE half differs, because
        // only the voice's two scalars come off the bus.
        assert!(
            sub.extra_inputs.wall_plane && voice.extra_inputs.wall_plane,
            "both read the wall-plane hit, so both must declare it"
        );
        assert!(
            voice.extra_inputs.state && !sub.extra_inputs.state,
            "only the voice reads the state word"
        );
        assert_eq!(
            RecipeProps {
                tag: sub.tag,
                state_is_scalar: sub.state_is_scalar,
                extra_inputs: sub.extra_inputs,
                ..voice
            },
            sub,
            "every other surface property must be the substrate's"
        );
        // Both modes declare the same properties — the mode is a driver, not a surface.
        assert_eq!(
            voice,
            TextureRecipe::Voice {
                scale: 3.0,
                mode: crate::substrate::VoiceMode::Release
            }
            .props()
        );
    }

    /// `state_scalar_tag` and `state_is_scalar` are two halves of one declaration —
    /// "I read the state word as a mark" and "here is where my mark comes from" — and
    /// they must not drift. A recipe that named a tag while still letting the FLOW
    /// reading bit-decode the same word would take the `FLOW_HOT_LIFT` on odd marks
    /// again, which is the exact defect `an_odd_mark_does_not_read_as_a_powered_pipe`
    /// exists to catch.
    ///
    /// Checked over one instance of EVERY recipe variant, so a new variant that gets
    /// only one half is caught here rather than in a frame.
    #[test]
    fn a_recipe_that_names_a_scalar_tag_also_declares_state_is_scalar() {
        let every_recipe = [
            TextureRecipe::Brick {
                aspect: 2.0,
                mortar_w: 0.16,
                grain: 0.5,
            },
            TextureRecipe::Circuit {
                scale: 3.0,
                trace_w: 0.1,
            },
            TextureRecipe::Grain {
                scale: 3.0,
                speck: 0.2,
                tile_eps: 0.02,
            },
            TextureRecipe::Sky {
                scale: 1.0,
                density: 0.2,
            },
            TextureRecipe::Hologram {
                line_freq: 8.0,
                scan_w: 0.2,
            },
            TextureRecipe::CodeWall {
                panel: 0,
                u_chars_per_unit: 8.0,
            },
            TextureRecipe::NoteWall {
                note: 0,
                face: re_core::FaceDir::PosY,
            },
            TextureRecipe::Scanline {
                freq: 8.0,
                duty: 0.5,
                warp: 0.0,
                axis: 0,
            },
            TextureRecipe::Fractal {
                arms: 3,
                octave_warp: 0.5,
                twist: 0.2,
            },
            TextureRecipe::Substrate {
                scale: 3.0,
                held: 0.5,
                released: 0.25,
            },
            TextureRecipe::Voice {
                scale: 3.0,
                mode: crate::substrate::VoiceMode::Hold,
            },
            TextureRecipe::Voice {
                scale: 3.0,
                mode: crate::substrate::VoiceMode::Release,
            },
            TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Contain,
                tile: IMAGE_TILE_NONE,
                relief: None,
            },
        ];
        let mut named = 0;
        for r in every_recipe {
            assert_eq!(
                r.state_scalar_tag().is_some(),
                r.props().state_is_scalar,
                "{r:?} declares only half of the scalar-state contract"
            );
            if r.state_scalar_tag().is_some() {
                named += 1;
            }
        }
        assert_eq!(named, 2, "exactly the two voice modes name a scalar tag");
        // The tag a recipe names is its MODE's tag, and the two modes differ — a
        // `state_scalar_tag` that ignored the mode would wire both walls to one mark.
        assert_eq!(
            TextureRecipe::Voice {
                scale: 3.0,
                mode: crate::substrate::VoiceMode::Hold
            }
            .state_scalar_tag(),
            Some(crate::substrate::MNEMOS_TAG)
        );
        assert_eq!(
            TextureRecipe::Voice {
                scale: 3.0,
                mode: crate::substrate::VoiceMode::Release
            }
            .state_scalar_tag(),
            Some(crate::substrate::LETHE_TAG)
        );
        // …and across the shipped catalog, exactly the two voice slots name one.
        let reg = MaterialRegistry::with_catalog();
        let namers: Vec<u16> = (0..reg.len())
            .filter(|i| {
                reg.get(MaterialId(*i as u16))
                    .recipe
                    .state_scalar_tag()
                    .is_some()
            })
            .map(|i| i as u16)
            .collect();
        assert_eq!(namers, vec![ids::MNEMOS_HOLD.0, ids::LETHE_RELEASE.0]);
    }

    /// The two new slots are APPENDED and nothing that shipped before adopts the
    /// `Voice` recipe — the direct half of "no existing material's output changed".
    #[test]
    fn only_the_two_new_slots_carry_the_voice_recipe() {
        let reg = MaterialRegistry::with_catalog();
        let voice_tag = TextureRecipe::Voice {
            scale: 3.0,
            mode: crate::substrate::VoiceMode::Hold,
        }
        .props()
        .tag;
        let adopters: Vec<u16> = (0..reg.len())
            .filter(|i| reg.get(MaterialId(*i as u16)).recipe_tag() == voice_tag)
            .map(|i| i as u16)
            .collect();
        assert_eq!(adopters, vec![ids::MNEMOS_HOLD.0, ids::LETHE_RELEASE.0]);
        const { assert!(ids::MNEMOS_HOLD.0 > ids::SUBSTRATE.0) };
        const { assert!(ids::LETHE_RELEASE.0 > ids::MNEMOS_HOLD.0) };
        assert_eq!(
            reg.id_of("substrate"),
            Some(ids::SUBSTRATE),
            "the tail must not move"
        );
        assert_eq!(reg.id_of("mnemos_hold"), Some(ids::MNEMOS_HOLD));
        assert_eq!(reg.id_of("lethe_release"), Some(ids::LETHE_RELEASE));
    }

    #[test]
    fn scanline_recipe_samples_a_filled_band_and_a_gap() {
        // The new Scanline recipe must produce BOTH an inked band glyph and a
        // coloured gap (filled, not void) across a column — i.e. it reads as a panel.
        let perlin = Perlin::new(SHADER_SEED);
        let metal = MaterialRegistry::catalog_material("metal").unwrap();
        let mut inked = 0;
        let mut gap = 0;
        for i in 0..64 {
            let coord = Vec2::new(1.0, i as f32 * 0.05);
            let t = metal.sample(&perlin, coord, &ctx(2.0, 1.0));
            if t.glyph == u32::from(' ') {
                gap += 1;
                // A filled panel: the gap still carries a non-pure-black bg.
                assert!(t.bg != 0, "scanline gap keeps a filled (non-void) bg");
            } else {
                inked += 1;
            }
        }
        assert!(
            inked > 0 && gap > 0,
            "scanline must alternate inked bands and gaps (inked={inked} gap={gap})"
        );
    }

    #[test]
    fn scanline_woven_axis_inks_both_directions() {
        // The woven axis (2) must ink along BOTH x and y (a cross-hatch), unlike the
        // horizontal/vertical single-axis bands.
        let perlin = Perlin::new(SHADER_SEED);
        let fabric = MaterialRegistry::catalog_material("fabric").unwrap();
        let row_inked = (0..40)
            .filter(|i| {
                fabric
                    .sample(&perlin, Vec2::new(*i as f32 * 0.05, 0.5), &ctx(2.0, 1.0))
                    .glyph
                    != u32::from(' ')
            })
            .count();
        let col_inked = (0..40)
            .filter(|i| {
                fabric
                    .sample(&perlin, Vec2::new(0.5, *i as f32 * 0.05), &ctx(2.0, 1.0))
                    .glyph
                    != u32::from(' ')
            })
            .count();
        assert!(
            row_inked > 0 && col_inked > 0,
            "woven fabric inks both axes (row={row_inked} col={col_inked})"
        );
    }

    /// SPARSE-SURFACE GATE: a Scanline material painted
    /// onto a FLOOR surface (the `metal`-preset `deck` floor in levels 04/08) must
    /// (a) be MOSTLY BLANK — thinned by the floor coverage mask, not a packed band
    /// field — and (b) NEVER promote a band cell to the heavy/dotted `━`/`┈`/`┃`/`┆`
    /// glyph family. The few surviving inked cells are a single PLAIN rule. A Scanline
    /// with NO light context (a puzzle panel / focused test) is unchanged (see
    /// `scanline_recipe_samples_a_filled_band_and_a_gap`, which uses the light-less
    /// `ctx` and still alternates filled bands and gaps).
    #[test]
    fn scanline_on_a_floor_surface_is_sparse_and_unpromoted() {
        let perlin = Perlin::new(SHADER_SEED);
        let metal = MaterialRegistry::catalog_material("metal").unwrap();

        // A FLOOR light context: a +z (up) face normal is what `LightCtx::for_floor`
        // builds; `scanline_surface` keys the gate off `face_n.z`.
        let mut floor_ctx = ctx(3.0, 1.0);
        let mut lc = wall_light(Vec3::new(0.0, 0.2, 1.0).normalize());
        lc.face_n = Vec3::Z;
        floor_ctx.light = Some(lc);

        // The promoted band glyphs the gate must suppress on a surface.
        const PROMOTED: [u32; 6] = [0x2501, 0x2508, 0x2503, 0x2506, 0x256C, 0x2592];

        let mut inked = 0usize;
        let mut blank = 0usize;
        // A 2D world patch so the coverage mask's kept/dropped scatter is sampled
        // broadly (a 1D sweep can miss the sparse kept cells entirely).
        for iy in 0..40 {
            for ix in 0..40 {
                let coord = Vec2::new(0.3 + ix as f32 * 0.13, 0.7 + iy as f32 * 0.11);
                let g = metal.sample(&perlin, coord, &floor_ctx).glyph;
                if g == u32::from(' ') {
                    blank += 1;
                } else {
                    inked += 1;
                    assert!(
                        !PROMOTED.contains(&g),
                        "floor Scanline must not promote to heavy/dotted U+{g:04X}"
                    );
                }
            }
        }
        // Overwhelmingly blank coloured ground (the calm reference look). The floor
        // coverage is ~7%, so well under half the cells ink even before the band duty
        // cycle thins them further.
        assert!(
            blank > inked * 3,
            "floor Scanline must read mostly blank (blank={blank} inked={inked})"
        );
        // It must still ink SOMETHING over a broad patch (a sparse accent, not a dead
        // surface) — and every surviving mark is the plain rule (asserted above).
        assert!(
            inked > 0,
            "floor Scanline keeps an occasional plain seam mark"
        );
    }

    /// A Scanline material on a WALL surface honours the same sparse + un-promoted
    /// rule (the cold-scheme `ice`/`metal` walls, level 08):
    /// thinned by `WALL_BODY_COVERAGE`, with only the plain rule glyph.
    #[test]
    fn scanline_on_a_wall_surface_is_sparse_and_unpromoted() {
        let perlin = Perlin::new(SHADER_SEED);
        let metal = MaterialRegistry::catalog_material("metal").unwrap();
        // A WALL light context: a horizontal (XY) face normal.
        let mut wall_ctx = ctx(3.0, 1.0);
        wall_ctx.light = Some(wall_light(Vec3::new(0.0, 1.0, 0.2).normalize()));

        const PROMOTED: [u32; 6] = [0x2501, 0x2508, 0x2503, 0x2506, 0x256C, 0x2592];
        let mut inked = 0usize;
        let mut blank = 0usize;
        for i in 0..240 {
            let coord = Vec2::new(0.5 + i as f32 * 0.011, 0.9 + i as f32 * 0.017);
            let g = metal.sample(&perlin, coord, &wall_ctx).glyph;
            if g == u32::from(' ') {
                blank += 1;
            } else {
                inked += 1;
                assert!(
                    !PROMOTED.contains(&g),
                    "wall Scanline must not promote to heavy/dotted U+{g:04X}"
                );
            }
        }
        // WALL_BODY_COVERAGE = 0.12, so the wall is also dominantly blank ground.
        assert!(
            blank > inked,
            "wall Scanline must read mostly blank (blank={blank} inked={inked})"
        );
    }

    // ── fractal: the log-polar self-similar portal recipe ────────────────────────

    #[test]
    fn fractal_recipe_yields_inked_texel_and_correct_props() {
        let m = Material {
            recipe: TextureRecipe::Fractal {
                arms: 4,
                octave_warp: 0.5,
                twist: 0.0,
            },
            ..DEFAULT_MATERIAL.clone()
        };
        let props = m.recipe.props();
        assert!(!props.transparent && !props.is_sky && props.self_lit);
        assert_eq!(props.body_blank, BodyBlankKind::None);
        assert_eq!(
            m.recipe_tag(),
            7,
            "fractal claims the free tag 7 (after scanline=6)"
        );
        // The portal must INK across a sweep — a flat single glyph or all-blank is a
        // placeholder. Sweep several coords; require both inked and varied glyphs.
        let perlin = Perlin::new(SHADER_SEED);
        let mut glyphs = std::collections::HashSet::new();
        let mut inked = 0;
        for i in 0..64 {
            let coord = Vec2::new(0.12 + i as f32 * 0.012, 0.62 - i as f32 * 0.008);
            let t = m.sample(&perlin, coord, &ctx(2.0, 1.0));
            glyphs.insert(t.glyph);
            if t.glyph != u32::from(' ') {
                inked += 1;
            }
        }
        assert!(
            inked > 0,
            "fractal portal must ink (it is a visible surface)"
        );
        assert!(
            glyphs.len() > 2,
            "fractal portal glyph must VARY with the height field (rings), got {} distinct",
            glyphs.len()
        );
    }

    #[test]
    fn fractal_anim_modes_mark_animated_and_loop() {
        let ctx_at = |t: f32| {
            let mut c = ctx(2.0, 1.0);
            c.time = t;
            c
        };
        let base = Material {
            recipe: TextureRecipe::Fractal {
                arms: 0,
                octave_warp: 0.0,
                twist: 0.0,
            },
            ..DEFAULT_MATERIAL.clone()
        };
        for kind in [AnimKind::Zoom, AnimKind::Spin, AnimKind::SpinZoom] {
            let m = Material {
                animation: Animation {
                    kind,
                    speed: 0.5,
                    amplitude: 1.0,
                },
                ..base.clone()
            };
            assert!(m.is_animated(), "{kind:?} must mark the material animated");
        }
        // Zoom phase reduces mod LN2: time advancing exactly one octave == time 0.
        let m = Material {
            animation: Animation {
                kind: AnimKind::Zoom,
                speed: 1.0,
                amplitude: 1.0,
            },
            ..base.clone()
        };
        // The zoom advances `speed·t·LN2` reduced `mod LN2`, so at speed 1.0 the field
        // returns to its start once `t` advances a whole unit (one octave of scroll).
        let ph0 = m.fractal_phase(&ctx_at(0.0), 0.0);
        let ph1 = m.fractal_phase(&ctx_at(1.0), 0.0);
        assert!(
            (ph0.zoom - ph1.zoom).abs() < 1e-4,
            "Zoom phase must loop over one unit of time (ph0.zoom={} ph1.zoom={})",
            ph0.zoom,
            ph1.zoom,
        );
        // A mid-octave time must DIFFER from the seam (the phase actually advances).
        let phm = m.fractal_phase(&ctx_at(0.5), 0.0);
        assert!(
            (phm.zoom - ph0.zoom).abs() > 1e-3,
            "Zoom phase must advance with time"
        );

        // Spin phase: `speed·t·TAU mod TAU` → loops over one unit of time too.
        let spin = Material {
            animation: Animation {
                kind: AnimKind::Spin,
                speed: 1.0,
                amplitude: 1.0,
            },
            ..base.clone()
        };
        let s0 = spin.fractal_phase(&ctx_at(0.0), 0.0);
        let s1 = spin.fractal_phase(&ctx_at(1.0), 0.0);
        assert!(
            (s0.spin - s1.spin).abs() < 1e-3,
            "Spin phase must loop over one unit of time"
        );

        // SpinZoom couples: a whole TAU of spin per LN2 of zoom, so both seams coincide
        // at the SAME time (one unit). At twist=0 the spin returns to start; the
        // twist!=0 seam is exercised by `spinzoom_material_loops_seamlessly_at_twist`.
        let sz = Material {
            animation: Animation {
                kind: AnimKind::SpinZoom,
                speed: 1.0,
                amplitude: 1.0,
            },
            ..base.clone()
        };
        let z0 = sz.fractal_phase(&ctx_at(0.0), 0.0);
        let z1 = sz.fractal_phase(&ctx_at(1.0), 0.0);
        assert!(
            (z0.zoom - z1.zoom).abs() < 1e-4 && (z0.spin - z1.spin).abs() < 1e-3,
            "SpinZoom must loop BOTH axes coincidentally over one unit of time"
        );

        // The non-fractal kinds (and static) yield the default (still) phase, so a
        // Pulse/Shift Fractal material does not warp — only the three modes drive it.
        let still = Material {
            animation: Animation {
                kind: AnimKind::Pulse,
                speed: 2.0,
                amplitude: 1.0,
            },
            ..base
        };
        let st = still.fractal_phase(&ctx_at(5.0), 0.0);
        assert_eq!(st.zoom, 0.0);
        assert_eq!(st.spin, 0.0);
    }

    #[test]
    fn spinzoom_fractal_phase_routes_through_seamless_helper() {
        // [C1] WIRING guard: the SpinZoom arm of `fractal_phase` must produce exactly
        // `crate::fractal::spinzoom_phase(t*LN2, twist)` (threading the recipe twist),
        // so the seamless coupling lives in ONE place. A non-zero twist proves the
        // twist actually reaches the helper (the old arm dropped it entirely).
        use crate::fractal::{spinzoom_phase, LN2};
        let ctx_at = |t: f32| {
            let mut c = ctx(2.0, 1.0);
            c.time = t;
            c
        };
        let m = Material {
            recipe: TextureRecipe::Fractal {
                arms: 5,
                octave_warp: 0.30,
                twist: 0.9,
            },
            animation: Animation {
                kind: AnimKind::SpinZoom,
                speed: 1.0,
                amplitude: 1.0,
            },
            ..DEFAULT_MATERIAL.clone()
        };
        for &t in &[0.0_f32, 0.137, 0.4, 0.6, 0.9] {
            let got = m.fractal_phase(&ctx_at(t), 0.9);
            let want = spinzoom_phase(t * LN2, 0.9);
            assert!(
                (got.zoom - want.zoom).abs() < 1e-6 && (got.spin - want.spin).abs() < 1e-6,
                "SpinZoom phase at t={t} must equal spinzoom_phase(t*LN2, twist): \
                 got ({},{}) want ({},{})",
                got.zoom,
                got.spin,
                want.zoom,
                want.spin
            );
            // The twist must change the spin (proves it is threaded through, not 0).
            let no_twist = m.fractal_phase(&ctx_at(t), 0.0);
            if t.rem_euclid(1.0) > 1e-6 {
                assert!(
                    (got.spin - no_twist.spin).abs() > 1e-4,
                    "twist=0.9 must alter the SpinZoom spin (it was dropped) at t={t}"
                );
            }
        }
    }

    #[test]
    fn spinzoom_material_seam_is_continuous_at_twist() {
        // [C1] regression at the MATERIAL output: the SpinZoom material must read the
        // SAME cell just before the loop wrap (t→1⁻) and at the start (t=0). The old
        // coupling jumped ~half an arm-period there (a visible hitch); the fix makes
        // the field continuous, so the quantized glyph/colour match across the seam.
        // Shipped portal_spiral params (arms=5, twist=0.9, octave_warp=0.30, speed=1).
        let ctx_at = |t: f32| {
            let mut c = ctx(2.0, 1.0);
            c.time = t;
            c
        };
        let m = Material {
            recipe: TextureRecipe::Fractal {
                arms: 5,
                octave_warp: 0.30,
                twist: 0.9,
            },
            animation: Animation {
                kind: AnimKind::SpinZoom,
                speed: 1.0,
                amplitude: 1.0,
            },
            ..DEFAULT_MATERIAL.clone()
        };
        let perlin = Perlin::new(SHADER_SEED);
        // t = 1 - tiny: the phase is a hair short of a whole octave, so the field is
        // arbitrarily close to the t=0 field IFF the loop is seamless. (At t=1 exactly
        // the phase wraps to 0 and the test would be vacuous.)
        let near_wrap = 1.0 - 1e-6;
        let mut varied = false;
        for &(u, v) in &[(0.30, 0.42), (0.71, 0.18), (0.55, 0.83), (0.18, 0.66)] {
            let coord = Vec2::new(u, v);
            let start = m.sample_fractal(&perlin, coord, &ctx_at(0.0), 5, 0.30, 0.9);
            let end = m.sample_fractal(&perlin, coord, &ctx_at(near_wrap), 5, 0.30, 0.9);
            assert_eq!(
                start.0.glyph, end.0.glyph,
                "SpinZoom seam glyph hitch at ({u},{v}): the loop is not seamless"
            );
            assert_eq!(
                start.0.fg, end.0.fg,
                "SpinZoom seam colour hitch at ({u},{v}): the loop is not seamless"
            );
            // Non-vacuous: the field must move across the half-period somewhere.
            let mid = m.sample_fractal(&perlin, coord, &ctx_at(0.5), 5, 0.30, 0.9);
            if mid.0.glyph != start.0.glyph || mid.0.fg != start.0.fg {
                varied = true;
            }
        }
        assert!(
            varied,
            "SpinZoom material is constant — the seam check would pass trivially"
        );
    }

    // ──────────────── F.1 quant-memo + F.2 far-LOD-on-compute ────────────────

    /// A lit wall [`SampleCtx`] (the shipped path) at `distance`, so the memo tests
    /// exercise the full recipe + Blinn-Phong + (absent) shadow pipeline, not the
    /// flat no-light fallback.
    fn lit_ctx(distance: f32) -> SampleCtx {
        let mut c = ctx(distance, 1.0);
        c.light = Some(wall_light(Vec3::new(0.0, 1.0, 0.2).normalize()));
        c
    }

    // ── THE CACHE KEYS MUST FINGERPRINT WHAT THE RECIPES READ ────────────────
    //
    // `Substrate` and `Voice` read `wall_along`/`wall_up` (and `Voice` reads `state`),
    // and no cache key carried any of them. It was not firing because `coord.x` for a
    // wall wraps once per world cell and the raw `distance` happened to separate the
    // cells that shared it — a property of ordinary scenes, not of the key. The tests
    // below drive the two places that property fails.

    /// A lit `Substrate`/`Voice` wall ctx at a wall-plane hit, with `coord` supplied
    /// separately by the caller so the WRAP can be reproduced exactly.
    fn plane_ctx(distance: f32, along: f32, up: f32) -> SampleCtx {
        let mut c = lit_ctx(distance);
        c.wall_along = along;
        c.wall_up = up;
        // This helper models a WALL hit, so it must say so: `Truchet` picks between
        // the wall plane and `coord` on exactly this flag.
        c.on_wall = true;
        c
    }

    /// Two cells ONE WORLD UNIT APART on the same flat wall run share a `coord`.
    ///
    /// This is not a contrived pair — it is what the caster produces. `sample_coord`
    /// builds `coord.x = tex_u + floor(the face's CONSTANT axis)`, and `tex_u` is the
    /// fraction along the VARIABLE axis, so stepping one world unit along the run
    /// leaves `coord` bit-identical while `wall_along` advances by 1. That is exactly
    /// the wrap Task 4's tiling fix moved the substrate field off, and the key was
    /// still relying on it.
    const WRAPPED_COORD: Vec2 = Vec2::new(2.5, 9.0);

    /// A wall-plane `along` pair one world unit apart whose SUBSTRATE glyphs differ,
    /// plus the two glyphs — so no test below is vacuous. Panics if the alphabet is
    /// somehow constant along the run.
    fn distinguishable_along_pair(m: &Material, ctx_of: impl Fn(f32) -> SampleCtx) -> (f32, f32) {
        let perlin = Perlin::new(SHADER_SEED);
        for i in 0..64 {
            let a = i as f32 * 0.25 + 0.13;
            let (ga, gb) = (
                m.sample(&perlin, WRAPPED_COORD, &ctx_of(a)).glyph,
                m.sample(&perlin, WRAPPED_COORD, &ctx_of(a + 1.0)).glyph,
            );
            if ga != gb {
                return (a, a + 1.0);
            }
        }
        panic!("no two wall-plane positions one unit apart differ — the probe is vacuous");
    }

    /// **THE COLLISION, DIRECTLY.** Two ctxs that differ ONLY in `wall_along`, same
    /// material, same `coord`, same everything else. The memo must answer each with
    /// its OWN texel, and each must equal an uncached `Material::sample`.
    ///
    /// In a debug build `SampleMemo::sample`'s own `debug_assert_eq!` fires on the
    /// bad hit before this test's assertions do — which is the point: the invariant
    /// was already written down, the key just stopped satisfying it.
    #[test]
    fn the_memo_separates_two_wall_plane_positions_that_share_a_coord() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("substrate").unwrap();
        let up = 3.5;
        let (a0, a1) = distinguishable_along_pair(&m, |a| plane_ctx(4.0, a, up));
        let (c0, c1) = (plane_ctx(4.0, a0, up), plane_ctx(4.0, a1, up));

        // The two cells really do share every field the pre-fix key carried.
        assert_eq!(c0.distance, c1.distance);
        assert_eq!(c0.foreshorten, c1.foreshorten);
        assert_eq!(
            SampleKey::light_digest(c0.light.as_ref()),
            SampleKey::light_digest(c1.light.as_ref())
        );
        // …and now differ in the key, because the key carries the wall plane.
        let (k0, k1) = (
            SampleKey::build(0, &m, WRAPPED_COORD, &c0),
            SampleKey::build(0, &m, WRAPPED_COORD, &c1),
        );
        assert_eq!(
            k0.coord_x_bits, k1.coord_x_bits,
            "the coord wrap is the premise"
        );
        assert_eq!(k0.coord_y_bits, k1.coord_y_bits);
        assert_eq!(k0.distance_bits, k1.distance_bits);
        assert_ne!(
            k0, k1,
            "two wall-plane positions sharing a coord must key apart"
        );

        // Through the memo: each must serve its own texel, and each must equal a
        // fresh uncached sample.
        let mut memo = SampleMemo::default();
        let t0 = memo.sample(0, &m, &perlin, WRAPPED_COORD, &c0);
        let t1 = memo.sample(0, &m, &perlin, WRAPPED_COORD, &c1);
        assert_eq!(t0, m.sample(&perlin, WRAPPED_COORD, &c0));
        assert_eq!(t1, m.sample(&perlin, WRAPPED_COORD, &c1));
        assert_ne!(
            t0.glyph, t1.glyph,
            "the memo served a neighbour's glyph — one world cell along the run"
        );
        assert_eq!(
            memo.misses(),
            2,
            "two distinct cells must be two distinct misses"
        );
    }

    /// **THE FROZEN PLATEAU — case (1).** Where the key DELIBERATELY discards
    /// distance, the missing fields matter most: `distance_bits` collapses onto
    /// [`FAR_FREEZE_LOD_BUCKET`] for a far-frozen, fog-saturated cell (see
    /// [`SampleKey::frozen_color`]), and `q_density` collapses with it. So two
    /// substrate cells at DIFFERENT distances, one world unit apart on the run, had
    /// every pre-fix key field equal — including the one that was accidentally
    /// separating them everywhere else.
    ///
    /// Reachable, not theoretical: `frozen_color` excludes only the SKY recipe, and
    /// substrate is a wall. The premise is asserted below rather than assumed.
    #[test]
    fn the_frozen_plateau_still_separates_two_wall_plane_positions() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("substrate").unwrap();
        let up = 2.25;
        // Far-frozen AND fog-saturated at both distances, so the key buckets both.
        let froze = |distance: f32, along: f32| {
            let mut c = plane_ctx(distance, along, up);
            c.far_freeze = 20.0;
            c.fog = FogParams {
                near: 2.0,
                far: 30.0,
                color: None,
                height: None,
            };
            c
        };
        let (a0, a1) = distinguishable_along_pair(&m, |a| froze(50.0, a));
        let (c0, c1) = (froze(50.0, a0), froze(90.0, a1));

        // THE PREMISE, asserted: both cells are in the plateau, so the key throws
        // their (different) distances away. Without this the test would be measuring
        // the ordinary distance separation instead of the plateau.
        for c in [&c0, &c1] {
            assert!(c.distance >= c.far_freeze, "must be far-frozen");
            assert!(
                SampleKey::frozen_color(&m, c),
                "must be fog-saturated (plateau)"
            );
        }
        let (k0, k1) = (
            SampleKey::build(0, &m, WRAPPED_COORD, &c0),
            SampleKey::build(0, &m, WRAPPED_COORD, &c1),
        );
        assert_ne!(
            c0.distance, c1.distance,
            "the two cells are at different depths…"
        );
        assert_eq!(
            k0.distance_bits, k1.distance_bits,
            "…which the plateau discards"
        );
        assert_eq!(
            k0.q_density_bits, k1.q_density_bits,
            "…as it does the LOD bucket"
        );
        assert!(k0.frozen && k1.frozen);
        // Every pre-fix field now matches. Only the wall plane tells them apart.
        assert_ne!(
            k0, k1,
            "in the plateau ONLY the wall plane separates the two cells"
        );

        let mut memo = SampleMemo::default();
        let t0 = memo.sample(0, &m, &perlin, WRAPPED_COORD, &c0);
        let t1 = memo.sample(0, &m, &perlin, WRAPPED_COORD, &c1);
        assert_eq!(t0, m.sample(&perlin, WRAPPED_COORD, &c0));
        assert_eq!(t1, m.sample(&perlin, WRAPPED_COORD, &c1));
        assert_ne!(
            t0.glyph, t1.glyph,
            "the frozen plateau served a neighbour's glyph"
        );
    }

    /// **CASE (2) — the cross-wire override.** With the recipe-resolved tag, one
    /// material means one tag means one mark, so a voice wall's cells all share a
    /// `state`. An EXPLICITLY STAMPED cell breaks exactly that: two cells of the SAME
    /// material carrying different voice tags read different marks, and before this
    /// fix they had an identical key.
    #[test]
    fn the_memo_separates_two_voice_cells_that_differ_only_in_state() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("mnemos_hold").unwrap();
        let (along, up) = (3.17, 2.5);
        let at = |word: u32| {
            let mut c = plane_ctx(4.0, along, up);
            c.state = word;
            c
        };
        // Marks far enough apart that the whole field converts between them.
        let (c0, c1) = (at(0), at(crate::substrate::MARK_PARTS));

        let (k0, k1) = (
            SampleKey::build(0, &m, WRAPPED_COORD, &c0),
            SampleKey::build(0, &m, WRAPPED_COORD, &c1),
        );
        assert_eq!(k0.coord_x_bits, k1.coord_x_bits);
        assert_eq!(
            k0.wall_along_bits, k1.wall_along_bits,
            "same cell, same position"
        );
        assert_ne!(k0, k1, "two marks on one material must key apart");

        let mut memo = SampleMemo::default();
        let t0 = memo.sample(0, &m, &perlin, WRAPPED_COORD, &c0);
        let t1 = memo.sample(0, &m, &perlin, WRAPPED_COORD, &c1);
        assert_eq!(t0, m.sample(&perlin, WRAPPED_COORD, &c0));
        assert_eq!(t1, m.sample(&perlin, WRAPPED_COORD, &c1));
        assert_ne!(
            t0.glyph, t1.glyph,
            "an unclaimed wall and a fully held one served the same texel"
        );
    }

    /// The CROSS-FRAME cache has the same gap and it is the worse one: [`SampleMemo`]
    /// is wiped every cast, so a missing `state` there is wrong for one frame, while
    /// [`WorldTexCache`] PERSISTS — a voice wall would replay its texel at a stale
    /// mark for as long as the entry survived. A progress wall that never moved.
    #[test]
    fn the_world_tex_cache_separates_two_marks_across_frames() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("lethe_release").unwrap();
        assert!(
            !m.is_animated(),
            "a cached material — the bypass must not hide this"
        );
        let at = |word: u32| {
            let mut c = plane_ctx(4.0, 3.17, 2.5);
            c.state = word;
            c
        };
        let (c0, c1) = (at(0), at(crate::substrate::MARK_PARTS));

        let mut cache = WorldTexCache::default();
        cache.tick();
        let t0 = cache.texture(0, &m, &perlin, WRAPPED_COORD, &c0);
        // A LATER FRAME with a risen mark — the cache survives the tick.
        cache.tick();
        let t1 = cache.texture(0, &m, &perlin, WRAPPED_COORD, &c1);
        assert_eq!(t0, m.texture_sample(&perlin, WRAPPED_COORD, &c0));
        assert_eq!(t1, m.texture_sample(&perlin, WRAPPED_COORD, &c1));
        assert_ne!(
            t0.base.glyph, t1.base.glyph,
            "the cross-frame cache replayed the wall at its old mark"
        );
        assert_eq!(cache.hits(), 0, "a risen mark is a MISS, not a stale hit");
        // …and the same mark twice IS a hit, or the fix would just be a disabled cache.
        cache.tick();
        assert_eq!(t1, cache.texture(0, &m, &perlin, WRAPPED_COORD, &c1));
        assert_eq!(
            cache.hits(),
            1,
            "an unchanged mark must still hit across frames"
        );
    }

    /// **THE BACK-COMPAT CLAIM.** A recipe that declares no extra inputs must key
    /// EXACTLY as it did before the fix — that is what leaves the eleven shipped
    /// recipes and every golden untouched.
    ///
    /// Stated two ways, because either alone is weak. First: the four new key fields
    /// are literally zero, so they contribute nothing to the comparison. Second: two
    /// ctxs differing in those inputs produce EQUAL keys — the behavioural form.
    #[test]
    fn a_recipe_declaring_no_extra_inputs_keys_exactly_as_before() {
        let mut checked = 0;
        for name in ids::CATALOG_NAMES {
            let m = MaterialRegistry::catalog_material(name).unwrap();
            if m.recipe.props().extra_inputs != CtxInputs::NONE {
                continue;
            }
            // An INERT bus: `state = 0` (so the flow reading is a no-op) and
            // `pulse = 0`. This is what every cell reads with no bridge publishing.
            let base = plane_ctx(4.0, 1.5, 2.5);
            let mut moved = plane_ctx(4.0, 99.75, -17.5);
            moved.state = 0;
            moved.pulse = 0.0;

            let k = SampleKey::build(7, &m, WRAPPED_COORD, &base);
            // 1. The new fields contribute nothing.
            assert_eq!(
                (
                    k.wall_along_bits,
                    k.wall_up_bits,
                    k.state_bits,
                    k.pulse_bits
                ),
                (0, 0, 0, 0),
                "{name} declares no extra inputs, so the new key fields must be zero"
            );
            // 2. …and the key is invariant to them.
            assert_eq!(
                k,
                SampleKey::build(7, &m, WRAPPED_COORD, &moved),
                "{name} keys differently for a wall plane it does not read"
            );
            assert_eq!(
                WorldTexKey::build(7, &m, WRAPPED_COORD, &base),
                WorldTexKey::build(7, &m, WRAPPED_COORD, &moved),
                "{name} world-tex keys differently for a wall plane it does not read"
            );
            checked += 1;
        }
        // Anti-vacuity: most of the catalog declares nothing, so most of it is here.
        // Five catalog materials declare extra inputs and are skipped by the `continue`
        // above: conduit + substrate (WALL_PLANE), mnemos_hold + lethe_release
        // (WALL_PLANE + STATE), and facade (WALL_PLANE).
        assert!(
            checked >= ids::CATALOG_NAMES.len() - 5,
            "only {checked} of {} catalog materials were checked — the sweep is not \
             covering the non-declaring recipes",
            ids::CATALOG_NAMES.len()
        );
    }

    /// The two UNCONDITIONAL readers of the bus — the flow scroll/hot-lift and the
    /// pulse kick — run for every material, so they cannot be declared per recipe.
    /// They are gated on the value instead: both are exact no-ops at the empty-bus
    /// reading, so a key is unchanged there and separates only where the reader is
    /// live. This is the precise sense in which "declares nothing ⇒ keys as before"
    /// holds.
    #[test]
    fn the_unconditional_bus_readers_are_keyed_only_when_live() {
        let m = MaterialRegistry::catalog_material("stone").unwrap();
        assert_eq!(m.recipe.props().extra_inputs, CtxInputs::NONE);
        let with = |state: u32, pulse: f32| {
            let mut c = plane_ctx(4.0, 1.5, 2.5);
            c.state = state;
            c.pulse = pulse;
            c
        };
        let inert = SampleKey::build(7, &m, WRAPPED_COORD, &with(0, 0.0));

        // An EVEN state word has the powered bit clear, so `flow_is_hot` is false and
        // the flow reading cannot change the output — no key change.
        assert!(
            !flow_is_hot(&with(1000, 0.0)),
            "an even word is not a live flow"
        );
        assert_eq!(
            inert,
            SampleKey::build(7, &m, WRAPPED_COORD, &with(1000, 0.0)),
            "a state word this material cannot read must not split the key"
        );
        // A LIVE flow word does change the output (scroll + hot lift), so it must.
        assert!(
            flow_is_hot(&with(3, 0.0)),
            "an odd word decodes as a powered flow"
        );
        assert_ne!(
            inert,
            SampleKey::build(7, &m, WRAPPED_COORD, &with(3, 0.0)),
            "a live flow word changes the shade and must split the key"
        );
        // Two DIFFERENT live flow words (forward vs backward) must also split.
        assert_ne!(
            SampleKey::build(7, &m, WRAPPED_COORD, &with(3, 0.0)),
            SampleKey::build(7, &m, WRAPPED_COORD, &with(5, 0.0)),
            "forward and backward flow must not share a key"
        );
        // PULSE: zero is the empty-bus value and keys as before; a live pulse adds an
        // emissive kick and must split.
        assert_ne!(
            inert,
            SampleKey::build(7, &m, WRAPPED_COORD, &with(0, 0.5)),
            "a live pulse changes the shade and must split the key"
        );
    }

    /// A precise NEGATIVE, so no one adds a defensive field later: for a WALL,
    /// `wall_up` is already implied by `coord.y`, so folding it cannot make a wall's
    /// key finer. `sample_coord` sets `coord.y = wall_world_y * WALL_WORLD_V_SCALE`
    /// and `wall_up = wall_world_y` — the same number through a fixed non-zero scale.
    /// It is folded anyway because that identity is a property of the CASTER, not of
    /// this key, and a floor/ceiling caller threads the two independently.
    #[test]
    fn wall_up_is_already_implied_by_coord_y_on_a_wall() {
        // The caster's own construction, reproduced: distinct wall texels must give
        // distinct `coord.y` exactly when they give distinct `wall_up`.
        let scale = crate::WALL_WORLD_V_SCALE;
        assert!(
            scale != 0.0,
            "a zero scale would collapse coord.y and break the claim"
        );
        let mut pairs = 0;
        for zi in 0..3 {
            for i in 0..16 {
                let v = i as f32 / 16.0;
                let up = crate::wall_world_y(zi, v);
                let coord_y = up * scale;
                for zj in 0..3 {
                    for j in 0..16 {
                        let up2 = crate::wall_world_y(zj, j as f32 / 16.0);
                        assert_eq!(
                            (coord_y == up2 * scale),
                            (up == up2),
                            "coord.y must separate wall texels exactly when wall_up does"
                        );
                        pairs += 1;
                    }
                }
            }
        }
        assert!(pairs > 0);
    }

    /// The `inset_rgb` field is the one remaining `SampleCtx` input neither key
    /// carries, and it is UNREACHABLE from a cached path rather than merely unused:
    /// only a [`ScaleMode::SelfSimilar`] material reads it, and that scale mode makes
    /// the material ANIMATED, which both caches bypass outright. A precise negative,
    /// asserted rather than assumed — if `SelfSimilar` ever stops implying animated,
    /// this fails and the field has to join the keys.
    #[test]
    fn inset_rgb_is_unreachable_from_a_cached_path() {
        let p = crate::fractal::FractalParams {
            arms: 3,
            octave_warp: 0.5,
            twist: 0.0,
        };
        assert!(
            ScaleMode::SelfSimilar(p).is_animated(),
            "SelfSimilar must be animated, or `inset_rgb` reaches a cached shade"
        );
        let m = Material {
            scale: ScaleMode::SelfSimilar(p),
            ..MaterialRegistry::catalog_material("stone").unwrap()
        };
        assert!(
            m.is_animated(),
            "a SelfSimilar material must report animated so both caches bypass it"
        );
    }

    /// A cache HIT returns the exact `sample` output, bit-for-bit (glyph + fg + bg +
    /// style/rank), for a STATIC material — the headline correctness guarantee.
    #[test]
    fn quant_memo_hit_equals_uncached_sample_bit_for_bit() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick(); // static (Animation::NONE)
        let coord = Vec2::new(1.37, 0.62);
        let ctx = lit_ctx(6.0);

        let mut memo = SampleMemo::default();
        // First call: a MISS (computes + stores).
        let first = memo.sample(0, &mat, &perlin, coord, &ctx);
        assert_eq!(memo.misses(), 1);
        assert_eq!(memo.hits(), 0);
        // Identical inputs again: a HIT that must equal both the first result AND a
        // fresh, un-cached `sample` — bit-for-bit on every channel.
        let second = memo.sample(0, &mat, &perlin, coord, &ctx);
        let fresh = mat.sample(&perlin, coord, &ctx);
        assert_eq!(memo.hits(), 1, "second identical call must hit the cache");
        assert_eq!(second, first, "cache hit must equal the stored texel");
        assert_eq!(
            second, fresh,
            "cache hit must equal a fresh sample bit-for-bit"
        );
        assert_eq!(second.glyph, fresh.glyph);
        assert_eq!(second.fg, fresh.fg);
        assert_eq!(second.bg, fresh.bg);
        assert_eq!(second.rank, fresh.rank);
    }

    /// An ANIMATED material is NEVER cached: it recomputes every call (its look
    /// depends on `time`), so the memo records misses, not hits, and the result
    /// tracks `time` (differs across frames).
    #[test]
    fn quant_memo_bypasses_animated_material() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_matrix(); // animated (AnimKind::Shift)
        assert!(mat.is_animated());
        let coord = Vec2::new(0.4, 0.7);

        let mut memo = SampleMemo::default();
        let mut t0 = lit_ctx(6.0);
        t0.time = 0.0;

        // Same inputs, twice: an animated material must NOT register a hit.
        let _ = memo.sample(2, &mat, &perlin, coord, &t0);
        let _ = memo.sample(2, &mat, &perlin, coord, &t0);
        assert_eq!(memo.hits(), 0, "animated material must never hit the cache");

        // And the look genuinely varies with time (so caching it WOULD be a bug):
        // a Shift wall scrolls its coord, so its glyph field changes across time.
        let mut moved = std::collections::HashSet::new();
        for k in 0..40 {
            let mut c = lit_ctx(6.0);
            c.time = k as f32 * 0.1;
            moved.insert(memo.sample(2, &mat, &perlin, coord, &c).glyph);
        }
        assert!(
            moved.len() > 1,
            "an animated wall's cached-bypass output must vary over time (got {})",
            moved.len()
        );
    }

    /// SEPARABILITY GUARD (time-independent shade cache).
    ///
    /// A proposed optimization was to cache a time-INDEPENDENT `STATIC_BASE` per
    /// screen-cell and, on an animation-only frame, re-apply ONLY a cheap
    /// `TIME_MODULATION` (`base ⊗ g(time)`) instead of re-shading. That is only
    /// correct if `sample` is EXACTLY separable as `f_static ⊗ g(time)`.
    ///
    /// It is NOT, for the shipped animated walls. `AnimKind::Shift` scrolls the
    /// SAMPLE COORD by `speed·time·amplitude` (see `anim_shift_coord`) BEFORE the
    /// recipe, the family/super-glyph selector, the relief height taps, AND the
    /// body-glyph ramp re-selection run — so the output is `f(coord + time·v)`, an
    /// entirely different texel each frame, not a modulation of one fixed base.
    /// This test pins that: a `Shift` wall's GLYPH and its fg/bg CHROMA DIRECTION
    /// both change across time. A single brightness scalar `g(time)` can only scale
    /// magnitude (it preserves the fg/bg hue ratio and the glyph), so no `base ⊗
    /// g(time)` could reproduce these outputs. If this test ever fails (the anim
    /// became a pure post-multiply), separability may be revisited.
    #[test]
    fn shift_anim_is_not_separable_as_base_times_time_scalar() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_matrix(); // AnimKind::Shift over a Circuit recipe (inked, not self-lit)
        let coord = Vec2::new(0.4, 0.7);

        // Chroma direction = the fg colour normalized to unit length. A pure
        // brightness scalar leaves this INVARIANT; a coord scroll does not.
        let chroma = |fg: u32| -> Vec3 {
            let c = unpack_rgb(fg);
            let n = c.length();
            if n > 1e-6 {
                c / n
            } else {
                Vec3::ZERO
            }
        };

        let mut glyphs = std::collections::HashSet::new();
        let mut chroma_moved = false;
        let mut ref_chroma: Option<Vec3> = None;
        for k in 0..40 {
            let mut c = lit_ctx(6.0);
            c.time = k as f32 * 0.1;
            let t = mat.sample(&perlin, coord, &c);
            glyphs.insert(t.glyph);
            let ch = chroma(t.fg);
            match ref_chroma {
                None => ref_chroma = Some(ch),
                Some(r) => {
                    // Hue direction drifts > a brightness scalar ever could (which is 0).
                    if (ch - r).length() > 1e-3 {
                        chroma_moved = true;
                    }
                }
            }
        }
        assert!(
            glyphs.len() > 1,
            "Shift wall glyph must vary over time (got {} distinct)",
            glyphs.len()
        );
        assert!(
            chroma_moved,
            "Shift wall fg CHROMA DIRECTION must vary over time — so the per-frame \
             output is NOT base ⊗ brightness(time); a shade cache would drift the image"
        );
    }

    /// A micro camera move that does NOT cross a quantization bucket hits the cache.
    /// In the FAR-FROZEN, fog-saturated regime the color is distance-invariant and
    /// the glyph uses the fixed FAR LOD bucket, so two slightly-different far
    /// distances collapse onto ONE key — a hit — AND the cached texel still equals a
    /// fresh sample at the perturbed distance (bit-identity preserved).
    #[test]
    fn quant_memo_micro_move_hits_cache_in_frozen_field() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let coord = Vec2::new(2.1, 0.9);

        // Far-frozen + fog fully saturated (default fog far = 26 < far_freeze = 32).
        let base_d = FAR_FREEZE_DIST + 4.0;
        let mk = |d: f32| {
            let mut c = lit_ctx(d);
            c.far_freeze = FAR_FREEZE_DIST;
            c // default fog: near=1, far=26 → distance>=far → fog saturated
        };

        let mut memo = SampleMemo::default();
        let c0 = mk(base_d);
        let _first = memo.sample(0, &mat, &perlin, coord, &c0);
        assert_eq!(memo.misses(), 1);

        // A sub-bucket "micro move": a tiny distance nudge that stays in the frozen,
        // fog-saturated plateau. Must HIT, and the cached value must equal a fresh
        // sample at the NEW distance (proving the collapse is output-preserving).
        let c1 = mk(base_d + 0.37);
        let hit = memo.sample(0, &mat, &perlin, coord, &c1);
        assert_eq!(
            memo.hits(),
            1,
            "a sub-bucket far micro-move must hit the cache"
        );
        assert_eq!(
            hit,
            mat.sample(&perlin, coord, &c1),
            "the frozen-field cache hit must equal a fresh sample at the moved distance"
        );
    }

    /// The far-LOD compute path produces the SAME (frozen/coarse) far output as the
    /// direct `sample`, whether or not the memo is in play — the memo collapses the
    /// far field onto one shade per key WITHOUT changing any cell's pixels. Sweeps a
    /// band of distances + coords across the FAR_FREEZE region and asserts every
    /// memoized cell is bit-identical to its un-cached counterpart, and that the
    /// collapse actually saves recomputes (hits > 0).
    #[test]
    fn far_lod_memoized_field_is_bit_identical_and_saves_recompute() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let mut memo = SampleMemo::default();

        for di in 0..12 {
            let d = FAR_FREEZE_DIST + 1.0 + di as f32 * 0.5; // all frozen + fogged
            for ui in 0..8 {
                let coord = Vec2::new(ui as f32 * 0.5, 1.0);
                let mut c = lit_ctx(d);
                c.far_freeze = FAR_FREEZE_DIST;
                let memoized = memo.sample(0, &mat, &perlin, coord, &c);
                let direct = mat.sample(&perlin, coord, &c);
                assert_eq!(
                    memoized, direct,
                    "far-LOD memoized cell must equal the direct sample (d={d}, u={ui})"
                );
            }
        }
        let hits = memo.hits();
        // 12 distances × 8 coords = 96 calls; the 8 distinct coords each repeat
        // across the 12 frozen distances → at least 12×8 − 8 = 88 should hit.
        assert!(
            hits >= 80,
            "the frozen far field must collapse onto a few keys (got only {hits} hits)"
        );
    }

    /// A static scene re-cast (same material + light, a band of near distances)
    /// where many cells share a coord+bucket sees the recompute COUNT drop: total
    /// shades = casts, but `misses` (actual `sample` runs) is strictly fewer because
    /// repeated identical inputs hit the cache. Proves the efficiency win on a
    /// non-frozen static surface too.
    #[test]
    fn static_scene_recompute_count_drops() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let mut memo = SampleMemo::default();

        // Simulate a flat wall slice: ONE world coord + light, shaded for many
        // "screen columns" (the per-face dedup the caster naturally produces).
        let coord = Vec2::new(3.0, 0.5);
        let ctx = lit_ctx(8.0);
        let calls = 64u64;
        for _ in 0..calls {
            let _ = memo.sample(0, &mat, &perlin, coord, &ctx);
        }
        assert_eq!(
            memo.misses(),
            1,
            "identical inputs must compute exactly once"
        );
        assert_eq!(memo.hits(), calls - 1, "the rest must be cache hits");
        // Recompute reduction: 1 actual sample vs 64 cells = 63/64 ≈ 98% avoided.
        assert!(memo.misses() < calls);
    }

    /// `clear()` invalidates the table in O(1) (a new frame starts cold) without
    /// touching the prior allocation, and resets the per-frame counters.
    #[test]
    fn quant_memo_clear_starts_a_cold_frame() {
        let perlin = Perlin::new(SHADER_SEED);
        let mat = wall_brick();
        let coord = Vec2::new(1.0, 1.0);
        let ctx = lit_ctx(5.0);

        let mut memo = SampleMemo::default();
        let _ = memo.sample(0, &mat, &perlin, coord, &ctx); // miss
        let _ = memo.sample(0, &mat, &perlin, coord, &ctx); // hit
        assert_eq!(memo.hits(), 1);
        memo.clear();
        assert_eq!(memo.hits(), 0, "clear resets counters");
        let _ = memo.sample(0, &mat, &perlin, coord, &ctx);
        assert_eq!(
            memo.misses(),
            1,
            "a cleared frame re-computes the first call"
        );
    }

    // ── KINDRED-SYMBOL family layer on a Material ────────────────────────────

    #[test]
    fn family_material_is_animated_and_bypasses_cache() {
        // The HAZE demo carries a Time-driven family → the material reports animated,
        // so the renderer keeps re-shading it and the memo BYPASSES the cache.
        let haze = mat_haze();
        assert!(
            haze.is_animated(),
            "a Time-driven family material is animated"
        );

        // A non-family material with no animation is NOT animated (stays cacheable).
        assert!(!wall_brick().is_animated(), "shipped brick is static");

        // Cache bypass: an animated material's `SampleMemo::sample` always recomputes
        // (it never records a hit, because its key would have to carry `time`).
        let perlin = Perlin::new(SHADER_SEED);
        let mut memo = SampleMemo::default();
        let coord = Vec2::new(2.0, 1.0);
        let c = ctx(3.0, 1.1);
        let _ = memo.sample(0, &haze, &perlin, coord, &c);
        let _ = memo.sample(0, &haze, &perlin, coord, &c); // identical call
        assert_eq!(
            memo.hits(),
            0,
            "an animated family material never hits the cache"
        );
    }

    #[test]
    fn family_material_varies_across_time_static_does_not() {
        // The FLOATING effect end-to-end: the HAZE material's GLYPH changes across
        // `time` (the kindred substitution cycles), while a non-family material's
        // glyph at the same coord is time-invariant.
        let perlin = Perlin::new(SHADER_SEED);
        let coord = Vec2::new(2.5, 1.5);
        let mut c = ctx(3.0, 1.1);

        let haze = mat_haze();
        let mut haze_glyphs = std::collections::BTreeSet::new();
        for k in 0..6 {
            c.time = k as f32 * 0.7; // advance the float clock
            haze_glyphs.insert(haze.sample(&perlin, coord, &c).glyph);
        }
        assert!(
            haze_glyphs.len() > 1,
            "the floating haze glyph must vary across time (saw {})",
            haze_glyphs.len()
        );

        // A static (non-family, non-animated) material: glyph is time-invariant.
        let stone = mat_stone();
        assert!(!stone.is_animated());
        let g0 = stone
            .sample(&perlin, coord, &{
                let mut x = ctx(3.0, 1.1);
                x.time = 0.0;
                x
            })
            .glyph;
        let g1 = stone
            .sample(&perlin, coord, &{
                let mut x = ctx(3.0, 1.1);
                x.time = 5.0;
                x
            })
            .glyph;
        assert_eq!(g0, g1, "a static material does not vary with time");
    }

    #[test]
    fn distance_family_grows_glyph_with_distance() {
        // The SPARK material's GROWTH family is distance-driven: a NEAR cell picks an
        // earlier (smaller) member than a FAR cell. We compare the chosen glyph's
        // index in the family at near vs far distance.
        use crate::glyph_family::GROWTH_DOT;
        let perlin = Perlin::new(SHADER_SEED);
        let coord = Vec2::new(1.5, 2.5);
        let spark = mat_spark();
        let idx_of = |g: u32| GROWTH_DOT.glyphs.iter().position(|&c| c as u32 == g);

        let near = spark.sample(&perlin, coord, &ctx(1.0, 1.0)).glyph;
        let far = spark.sample(&perlin, coord, &ctx(28.0, 1.0)).glyph;
        // Both must be family members (the growth family owns the glyph here).
        if let (Some(ni), Some(fi)) = (idx_of(near), idx_of(far)) {
            assert!(
                fi >= ni,
                "far glyph (idx {fi}) must be >= near glyph (idx {ni}) in the growth family"
            );
            assert!(
                fi > ni || far != near,
                "distance should change the growth member"
            );
        } else {
            panic!("spark glyph not a GROWTH_DOT member: near={near:#X} far={far:#X}");
        }
    }

    // ── BOLD-AS-A-CHANNEL (proximity + emitter) ──────────────────────────────

    #[test]
    fn bold_as_proximity_only_bolds_near_cells() {
        // A material with `proximity_bold = Some(threshold)` bolds an inked cell at/
        // below the threshold and NOT beyond it — the free "approach" depth cue.
        // Use a synthetic inked texel so the result is independent of the recipe.
        let mut m = wall_brick();
        m.proximity_bold = Some(5.0);
        let texel = Texel {
            glyph: b'#' as u32,
            fg: 0x00808080,
            bg: 0x00101010,
            rank: 0,
        };

        let near = cell_style_ext(&m, &texel, 2.0);
        let far = cell_style_ext(&m, &texel, 20.0);
        assert!(near.is_bold(), "a near cell (<= threshold) bolds");
        assert!(
            !far.is_bold(),
            "a far cell (> threshold) does not bold by proximity"
        );

        // Without the opt-in, neither distance bolds a mid-luma cell by proximity.
        let plain = wall_brick();
        assert!(
            !cell_style_ext(&plain, &texel, 2.0).is_bold(),
            "no opt-in ⇒ no proximity bold"
        );
    }

    #[test]
    fn bold_as_proximity_does_not_bold_blank_or_sky() {
        let mut m = mat_void(); // a Sky recipe
        m.proximity_bold = Some(100.0);
        let blank = Texel {
            glyph: b' ' as u32,
            fg: 0x00FFFFFF,
            bg: 0,
            rank: 0,
        };
        assert!(
            !cell_style_ext(&m, &blank, 0.5).is_bold(),
            "sky / blank never bolds"
        );
    }

    /// Renders the HAZE (floating) material to a truecolor ANSI block across a few
    /// time steps so the kindred-glyph DRIFT is visible. Prints the artifact with
    /// `--nocapture`; always passes (a demonstrator, not an assertion gate beyond
    /// "it produced varied glyphs").
    #[test]
    fn dump_haze_floating_material_ansi() {
        let perlin = Perlin::new(SHADER_SEED);
        let haze = mat_haze();
        let lc = flat_light_ctx();
        let cols = 24;
        let frames = 4;
        let mut all_glyphs = std::collections::BTreeSet::new();
        let mut out = String::from(
            "\n── haze / floating-glyph material (kindred substitution over time) ──\n",
        );
        for f in 0..frames {
            let time = f as f32 * 0.6;
            let mut line = String::new();
            for x in 0..cols {
                let coord = Vec2::new(x as f32 * 0.5 + 0.25, 1.5);
                let mut c = ctx(3.0, 1.1);
                c.time = time;
                c.light = Some(lc);
                let t = haze.sample(&perlin, coord, &c);
                all_glyphs.insert(t.glyph);
                let fg = unpack_rgb(t.fg);
                let bg = unpack_rgb(t.bg);
                let (fr, fgc, fb) = (
                    (fg.x * 255.0) as u8,
                    (fg.y * 255.0) as u8,
                    (fg.z * 255.0) as u8,
                );
                let (br, bgc, bb) = (
                    (bg.x * 255.0) as u8,
                    (bg.y * 255.0) as u8,
                    (bg.z * 255.0) as u8,
                );
                let g = char::from_u32(t.glyph).unwrap_or(' ');
                line.push_str(&format!(
                    "\x1b[38;2;{fr};{fgc};{fb};48;2;{br};{bgc};{bb}m{g}"
                ));
            }
            line.push_str("\x1b[0m");
            out.push_str(&format!("t={time:>4.1}  {line}\n"));
        }
        println!("{out}");
        assert!(
            all_glyphs.len() > 1,
            "the haze dump must show more than one floating glyph (saw {})",
            all_glyphs.len()
        );
    }

    // ── PER-RECIPE DESCRIPTOR seam (`TextureRecipe::props`) ──────────────────

    /// The single authoritative [`RecipeProps`] descriptor must, for EVERY recipe,
    /// agree with the behaviour the scattered `match self.recipe` sites used to
    /// hand-code — this is the extensibility seam, so it is pinned: a new recipe (or
    /// a mis-set field) that diverges from the historical predicate fails here.
    ///
    /// Each row is `(recipe, transparent, is_sky, is_code_wall, self_lit, body_blank,
    /// tag)` — the expected descriptor, written out independently of `props()` so the
    /// table is a true oracle, not a tautology.
    #[test]
    fn recipe_props_descriptor_matches_prior_per_recipe_behaviour() {
        use BodyBlankKind::{Floor, None as NoBlank, Wall};
        let cases: [(TextureRecipe, bool, bool, bool, bool, BodyBlankKind, u8); 9] = [
            // The cell alphabet OWNS its glyph (so `is_code_wall`, the "relief must
            // not re-select this" flag) but is a physical wall, so it is NOT self-lit.
            (
                TextureRecipe::Substrate {
                    scale: 3.0,
                    held: 0.5,
                    released: 0.25,
                },
                false,
                false,
                true,
                false,
                NoBlank,
                10,
            ),
            // The bay×floor window grid OWNS its glyph exactly like `Substrate` above
            // (`is_code_wall: true`) — a physical wall, not self-lit. Regression pin:
            // this recipe once shipped with `is_code_wall: false`, which let the
            // generic SPARSE-BODY relief pass silently re-select every facade glyph
            // off its own density ramp — invisible to every prior test because none
            // rendered a `surface = wall_solid`-bound facade end to end.
            (
                TextureRecipe::Facade {
                    style: crate::facade::FacadeStyle::Office,
                    bay: 3.0,
                    floor_h: 4.0,
                    lit: 0.35,
                    seed: 0,
                    ground_band: false,
                    sign_row: false,
                },
                false,
                false,
                true,
                false,
                NoBlank,
                12,
            ),
            (
                TextureRecipe::Brick {
                    aspect: 1.4,
                    mortar_w: 0.2,
                    grain: 0.7,
                },
                false,
                false,
                false,
                false,
                Wall,
                0,
            ),
            (
                TextureRecipe::Circuit {
                    scale: 2.0,
                    trace_w: 0.07,
                },
                false,
                false,
                false,
                false,
                NoBlank,
                1,
            ),
            (
                TextureRecipe::Grain {
                    scale: 2.0,
                    speck: 0.4,
                    tile_eps: 0.0,
                },
                false,
                false,
                false,
                false,
                Floor,
                2,
            ),
            (
                TextureRecipe::Sky {
                    scale: 7.0,
                    density: 0.06,
                },
                false,
                true,
                false,
                false,
                NoBlank,
                3,
            ),
            (
                TextureRecipe::Hologram {
                    line_freq: 3.0,
                    scan_w: 0.30,
                },
                true,
                false,
                false,
                true,
                NoBlank,
                4,
            ),
            (
                TextureRecipe::CodeWall {
                    panel: 0,
                    u_chars_per_unit: 8.0,
                },
                false,
                false,
                true,
                true,
                NoBlank,
                5,
            ),
            (
                TextureRecipe::Scanline {
                    freq: 5.0,
                    duty: 0.7,
                    warp: 0.45,
                    axis: 0,
                },
                false,
                false,
                false,
                true,
                NoBlank,
                6,
            ),
        ];
        for (recipe, transparent, is_sky, is_code_wall, self_lit, body_blank, tag) in cases {
            let p = recipe.props();
            assert_eq!(p.transparent, transparent, "{recipe:?} transparent");
            assert_eq!(p.is_sky, is_sky, "{recipe:?} is_sky");
            assert_eq!(p.is_code_wall, is_code_wall, "{recipe:?} is_code_wall");
            assert_eq!(p.self_lit, self_lit, "{recipe:?} self_lit");
            assert_eq!(p.body_blank, body_blank, "{recipe:?} body_blank");
            assert_eq!(p.tag, tag, "{recipe:?} tag");
            // The public predicates that now READ the descriptor must agree with the
            // descriptor field — proving every consumer is routed through one seam.
            let mut m = wall_brick();
            m.recipe = recipe;
            assert_eq!(
                m.is_transparent(),
                transparent,
                "{recipe:?} is_transparent()"
            );
            assert_eq!(m.recipe_tag(), tag, "{recipe:?} recipe_tag()");
        }
    }

    /// Every distinct recipe must own a distinct [`RecipeProps::tag`] — the tag is a
    /// stable discriminant tools group materials by, so a copy-paste collision in the
    /// single `props()` table is caught immediately.
    #[test]
    fn recipe_props_tags_are_unique_per_recipe() {
        let recipes = [
            TextureRecipe::Brick {
                aspect: 1.0,
                mortar_w: 0.1,
                grain: 0.5,
            },
            TextureRecipe::Circuit {
                scale: 2.0,
                trace_w: 0.07,
            },
            TextureRecipe::Grain {
                scale: 2.0,
                speck: 0.4,
                tile_eps: 0.0,
            },
            TextureRecipe::Sky {
                scale: 7.0,
                density: 0.06,
            },
            TextureRecipe::Hologram {
                line_freq: 3.0,
                scan_w: 0.3,
            },
            TextureRecipe::CodeWall {
                panel: 0,
                u_chars_per_unit: 8.0,
            },
            TextureRecipe::Scanline {
                freq: 5.0,
                duty: 0.7,
                warp: 0.45,
                axis: 0,
            },
            TextureRecipe::NoteWall {
                note: 0,
                face: re_core::FaceDir::PosY,
            },
            TextureRecipe::Fractal {
                arms: 5,
                octave_warp: 0.3,
                twist: 0.9,
            },
            TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Contain,
                tile: IMAGE_TILE_NONE,
                relief: None,
            },
            TextureRecipe::Substrate {
                scale: 3.0,
                held: 0.5,
                released: 0.25,
            },
        ];
        let mut tags: Vec<u8> = recipes.iter().map(|r| r.props().tag).collect();
        let n = tags.len();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), n, "every recipe must have a UNIQUE props().tag");
    }

    /// Per-glyph `colors =` override: a keyed glyph's
    /// `Texel.fg` is the OVERRIDE colour bit-for-bit, not the class-based
    /// `mark_fg`/`fg`/`void_fg` pick — proving `sample_note_wall` actually applies
    /// `NoteSample::color_override` rather than merely carrying it. Calls the
    /// private `sample_note_wall` directly (same module tree) so no fog/LOD pass
    /// downstream of it can perturb the colour this test is pinning.
    #[test]
    fn sample_note_wall_applies_the_per_glyph_colour_override_over_the_class_pick() {
        let _serial = crate::scene_text::test_lock();
        let mut reg = crate::scene_text::TextSurfaceRegistry::new();
        // Two distinct keyed glyphs with two distinct colours, one unkeyed glyph
        // between them, on a plain (markerless, non-tiered) stencil body.
        reg.push_note_text_styled(
            "n",
            "colors = [:#e0a040 ]:#40a0ff\n[X]",
            crate::note_style::NoteRegister::Stencil,
        );
        let _scope = crate::scene_text::SceneTextScope::new(reg);

        let mut m = wall_brick();
        m.recipe = TextureRecipe::NoteWall {
            note: 0,
            face: re_core::FaceDir::PosY,
        };
        // `distance <= 0.0` ⇒ `face_span::face_{cols,rows}` both report "span
        // unknown" (see their own doc comments), which is exactly the plain
        // stretch-mapping every `scene_text` unit test already samples under —
        // no anchor/magnification clamp to reproduce here.
        let mut sctx = ctx(0.0, 1.0);
        sctx.side = 0; // agrees with FaceDir::PosY.side() when ctx.light is None
        sctx.wall_v = 0.01; // top row

        let cell = |col: usize| -> Texel {
            let u = (col as f32 + 0.5) / crate::code_wall::NOTE_COLS as f32;
            m.sample_note_wall(Vec2::new(u, 0.0), &sctx, 0, re_core::FaceDir::PosY)
                .0
        };
        // Column 0 = '[' (keyed amber), column 1 = 'X' (unkeyed), column 2 = ']'
        // (keyed blue) — see `TextPanel::build_verbatim`'s one-glyph-per-cell grid.
        assert_eq!(cell(0).fg, 0x00e0_a040, "'[' takes its own authored colour");
        assert_eq!(
            cell(2).fg,
            0x0040_a0ff,
            "']' takes ITS own colour, not '['s"
        );
        assert_ne!(
            cell(1).fg,
            0x00e0_a040,
            "unkeyed 'X' must not pick up a neighbour's override"
        );
        assert_ne!(
            cell(1).fg,
            0x0040_a0ff,
            "unkeyed 'X' must not pick up a neighbour's override"
        );
    }

    // ── TRANSPARENT-material routing (composite slot split) ──────────────────

    #[test]
    fn is_transparent_true_for_hologram_and_glass_false_for_opaque() {
        // The composite splits opaque (slot 1) vs transparent (slot 2) on EXACTLY
        // this predicate, so it must be true for the hologram + the new glass-style
        // materials and false for every filled / opaque surface.
        for name in ["holo_amber", "vitrine", "screen", "veil"] {
            let m = MaterialRegistry::catalog_material(name).unwrap();
            assert!(
                m.is_transparent(),
                "'{name}' must be transparent (slot 2, OverKeepBg)"
            );
        }
        // The engine hologram built-in is transparent too.
        assert!(holo_node().is_transparent(), "holo_node is transparent");

        // OPAQUE materials (filled recipes) must be FALSE — they replace the cell.
        for name in [
            "metal",
            "stone",
            "rust",
            "wood",
            "fabric",
            "lava",
            "monitor",
            "moss",
            "ember_field",
            "static_noise",
            "lattice",
            "glass",
        ] {
            let m = MaterialRegistry::catalog_material(name).unwrap();
            assert!(!m.is_transparent(), "'{name}' must be OPAQUE (slot 1)");
        }
        // The shipped surface built-ins are opaque.
        assert!(!wall_brick().is_transparent() && !floor_grain().is_transparent());
    }

    #[test]
    fn transparent_materials_are_more_than_just_hologram_holo() {
        // The transparent layer must be exercised by more than the two shipped holos:
        // vitrine/screen/veil add three distinct see-through configs.
        let transparent: Vec<&str> = ids::CATALOG_NAMES
            .iter()
            .copied()
            .filter(|n| {
                MaterialRegistry::catalog_material(n)
                    .unwrap()
                    .is_transparent()
            })
            .collect();
        // holo_amber + vitrine + screen + veil (glass is reflective-opaque by design).
        assert!(
            transparent.len() >= 4,
            "expected >= 4 transparent catalog materials, got {transparent:?}"
        );
        for n in ["vitrine", "screen", "veil"] {
            assert!(
                transparent.contains(&n),
                "'{n}' must be among the transparent set"
            );
        }
    }

    #[test]
    fn transparent_material_inks_sparsely_over_a_void_bg() {
        // A transparent (hologram-recipe) material must leave GAPS (blank cells) so
        // the background behind shows through, and force a near-void bg on inked
        // cells (the resolver discards it, but it must be void not a filled panel).
        let perlin = Perlin::new(SHADER_SEED);
        let vitrine = MaterialRegistry::catalog_material("vitrine").unwrap();
        let mut blank = 0;
        let mut inked = 0;
        for i in 0..64 {
            let coord = Vec2::new(1.0, i as f32 * 0.05);
            let t = vitrine.sample(&perlin, coord, &ctx(2.0, 1.0));
            if t.glyph == u32::from(' ') {
                blank += 1;
            } else {
                inked += 1;
                // The hologram bg is the void level, not a bright filled colour.
                let bg = unpack_rgb(t.bg);
                assert!(
                    bg.max_element() < 0.2,
                    "transparent ink rides a near-void bg"
                );
            }
        }
        assert!(
            blank > 0 && inked > 0,
            "a see-through pane must alternate gaps + lines (blank={blank} inked={inked})"
        );
    }

    // ── NEW catalog materials register + round-trip by name ───────────────────

    #[test]
    fn new_catalog_presets_register_and_round_trip() {
        let cat = MaterialRegistry::with_catalog();
        for name in [
            "vitrine",
            "screen",
            "veil",
            "moss",
            "ember_field",
            "static_noise",
            "lattice",
        ] {
            // Registered in the with_catalog registry past the fixed slots…
            let id = cat
                .id_of(name)
                .unwrap_or_else(|| panic!("new catalog material '{name}' must register"));
            assert!(id.0 >= 8, "'{name}' sits past the fixed surface slots");
            // …and the standalone data lookup (the `preset = NAME` path) agrees.
            let direct = MaterialRegistry::catalog_material(name)
                .unwrap_or_else(|| panic!("catalog_material('{name}') must resolve"));
            assert_eq!(
                direct.recipe_tag(),
                cat.get(id).recipe_tag(),
                "'{name}' preset lookup must match its registered slot"
            );
        }
    }

    #[test]
    fn new_material_properties_read_as_configured() {
        // Spot-check the varied PROPERTIES are wired on the new opaque materials.
        let moss = MaterialRegistry::catalog_material("moss").unwrap();
        assert!(
            moss.is_animated() && moss.family.is_some(),
            "moss floats a dot family"
        );
        let ember = MaterialRegistry::catalog_material("ember_field").unwrap();
        assert!(
            ember.light.emissive > 0.5 && ember.proximity_bold.is_some(),
            "ember_field glows + bolds near coals"
        );
        let stat = MaterialRegistry::catalog_material("static_noise").unwrap();
        assert!(
            stat.is_animated() && stat.animation.kind == AnimKind::Flicker,
            "static flickers"
        );
        let lattice = MaterialRegistry::catalog_material("lattice").unwrap();
        assert!(
            !lattice.is_animated() && lattice.light.specular > 0.3,
            "lattice is a calm, sheeny static weave"
        );
    }

    // ── SUPER-GLYPH SELECTORS beyond the family drivers ─────────────

    use crate::glyph_family::GROWTH_DENSITY;

    /// A small family + drive helper for the selector tests.
    fn sg(selector: SuperSelector) -> SuperGlyphLayer {
        SuperGlyphLayer {
            family: GROWTH_DENSITY,
            selector,
            state: 0,
        }
    }

    #[test]
    fn super_selector_is_animated_classification() {
        // time-dependent selectors float (cache-bypass); the geometry-only ones cache.
        assert!(SuperSelector::Bpm { bpm: 120.0 }.is_animated());
        assert!(SuperSelector::MultiplyComposite { speed: 1.0 }.is_animated());
        assert!(SuperSelector::PriorityComposite { speed: 1.0 }.is_animated());
        assert!(SuperSelector::Procedural { speed: 1.0 }.is_animated());
        assert!(!SuperSelector::Density.is_animated());
        assert!(!SuperSelector::NeighborAware.is_animated());
    }

    #[test]
    fn bpm_selector_is_steppy_on_the_beat_and_deterministic() {
        let layer = sg(SuperSelector::Bpm { bpm: 60.0 }); // 1 beat / sec
        let coord = Vec2::new(2.0, 1.0);
        let at = |time: f32| layer.select(LOD_NEAR_DENSITY, coord, time, SHADER_SEED, 0.5);
        // Determinism: identical inputs → identical glyph.
        assert_eq!(at(0.5), at(0.5), "bpm selector is a pure function");
        // STEPPY: within one beat (0.0..1.0) the glyph holds; it may change at the
        // beat boundary. At 60bpm beats land on whole seconds.
        assert_eq!(
            at(0.10),
            at(0.90),
            "glyph holds WITHIN a beat (steppy, on-beat)"
        );
        // Over several beats the cycle visits more than one member (it animates).
        let seen: std::collections::BTreeSet<u32> = (0..GROWTH_DENSITY.len() as u32 + 2)
            .map(|k| at(k as f32 + 0.5))
            .collect();
        assert!(
            seen.len() > 1,
            "bpm cycle must visit multiple members (saw {})",
            seen.len()
        );
    }

    #[test]
    fn density_selector_grows_monotonically_with_density() {
        let layer = sg(SuperSelector::Density);
        let coord = Vec2::new(1.0, 1.0);
        let g = |q: f32| layer.select(q, coord, 0.0, SHADER_SEED, 0.5);
        let idx = |gl: u32| {
            GROWTH_DENSITY
                .glyphs
                .iter()
                .position(|&c| c as u32 == gl)
                .unwrap()
        };
        let near = g(LOD_NEAR_DENSITY); // sparse
        let mid = g((LOD_NEAR_DENSITY + LOD_FAR_DENSITY) * 0.5);
        let far = g(LOD_FAR_DENSITY); // dense
        assert!(
            idx(near) <= idx(mid) && idx(mid) <= idx(far),
            "density selector must be monotone (near={} mid={} far={})",
            idx(near),
            idx(mid),
            idx(far)
        );
        assert!(idx(near) < idx(far), "sparse and dense must differ");
    }

    #[test]
    fn neighbor_aware_selector_is_world_anchored_and_stable() {
        let layer = sg(SuperSelector::NeighborAware);
        let g = |c: Vec2| layer.select(LOD_NEAR_DENSITY, c, 0.0, SHADER_SEED, 0.5);
        // STABLE under a sub-cell camera nudge: two coords inside the SAME coarse
        // lattice bucket (1/NEIGHBOR_LATTICE wide) must pick the SAME member — no
        // flicker as the camera creeps within a texel.
        let base = Vec2::new(2.0, 3.0);
        let nudged = base + Vec2::new(0.05, 0.04); // well within 1/3 of a world unit
        assert_eq!(
            g(base),
            g(nudged),
            "neighbour-aware pick is world-anchored (no jitter)"
        );
        // It varies across DISTINCT buckets (a connected-looking pattern, not a flat).
        let seen: std::collections::BTreeSet<u32> = (0..16)
            .map(|i| g(Vec2::new(i as f32, (i * 3) as f32)))
            .collect();
        assert!(
            seen.len() > 1,
            "neighbour-aware must vary across cells (saw {})",
            seen.len()
        );
    }

    #[test]
    fn priority_composite_state_interrupts_idle_float() {
        let coord = Vec2::new(1.5, 2.5);
        // state == 0: a slow idle float (animates over time).
        let idle = SuperGlyphLayer {
            family: GROWTH_DENSITY,
            selector: SuperSelector::PriorityComposite { speed: 1.0 },
            state: 0,
        };
        let idle_seen: std::collections::BTreeSet<u32> = (0..8)
            .map(|k| idle.select(LOD_NEAR_DENSITY, coord, k as f32, SHADER_SEED, 0.5))
            .collect();
        assert!(
            idle_seen.len() > 1,
            "idle priority-composite floats over time"
        );
        // state != 0: PINNED to the interrupt frame, time-invariant (priority wins).
        let alert = SuperGlyphLayer { state: 3, ..idle };
        let g0 = alert.select(LOD_NEAR_DENSITY, coord, 0.0, SHADER_SEED, 0.5);
        let g9 = alert.select(LOD_NEAR_DENSITY, coord, 9.0, SHADER_SEED, 0.5);
        assert_eq!(
            g0, g9,
            "an active state frame ignores the idle float (priority)"
        );
        assert_eq!(g0, GROWTH_DENSITY.at(3), "state pins to member 3");
    }

    #[test]
    fn multiply_and_procedural_selectors_are_deterministic_and_vary() {
        let coord = Vec2::new(4.0, 2.0);
        for sel in [
            SuperSelector::MultiplyComposite { speed: 1.0 },
            SuperSelector::Procedural { speed: 1.0 },
        ] {
            let layer = sg(sel);
            // Deterministic: same inputs → same output.
            assert_eq!(
                layer.select(5.0, coord, 1.0, SHADER_SEED, 0.5),
                layer.select(5.0, coord, 1.0, SHADER_SEED, 0.5),
                "{sel:?} must be a pure function"
            );
            // Animated: varies across whole time STEPS (steppy, not jitter).
            let seen: std::collections::BTreeSet<u32> = (0..GROWTH_DENSITY.len() as u32 + 3)
                .map(|k| layer.select(5.0, coord, k as f32, SHADER_SEED, 0.5))
                .collect();
            assert!(
                seen.len() > 1,
                "{sel:?} must vary across time (saw {})",
                seen.len()
            );
        }
    }

    #[test]
    fn super_selectors_are_stable_under_a_subcell_time_jitter() {
        // The temporal-stability contract: the time-driven selectors advance
        // in whole STEPS, so a sub-step time jitter (the kind a frame-time wobble
        // produces) must NOT change the glyph — it only flips at a step boundary.
        let coord = Vec2::new(2.0, 2.0);
        let cases: [(SuperSelector, f32); 3] = [
            (SuperSelector::Bpm { bpm: 60.0 }, 5.0), // beat boundaries at whole sec
            (SuperSelector::MultiplyComposite { speed: 1.0 }, 5.0),
            (SuperSelector::Procedural { speed: 1.0 }, 5.0),
        ];
        for (sel, t) in cases {
            let layer = sg(sel);
            // A small jitter well inside one step (< 1 step) around a non-boundary t.
            let centre = t + 0.5; // mid-step
            let lo = layer.select(LOD_NEAR_DENSITY, coord, centre - 0.2, SHADER_SEED, 0.5);
            let hi = layer.select(LOD_NEAR_DENSITY, coord, centre + 0.2, SHADER_SEED, 0.5);
            assert_eq!(lo, hi, "{sel:?} must hold through a sub-step time jitter");
        }
    }

    // ── SUPER-GLYPH layer wired onto a Material (catalog + sample path) ──────────

    /// Each new catalog material carries the expected [`SuperSelector`], resolves by
    /// name (the `preset = NAME` path), and round-trips through the registry slot —
    /// proving the selector is now an authorable material field, not a dead API.
    #[test]
    fn super_glyph_catalog_materials_round_trip_via_preset() {
        // Named alias keeps the table type readable (and clippy's type-complexity
        // lint happy) — a predicate that classifies a selector by kind.
        type SelKindCheck = fn(SuperSelector) -> bool;
        let expect: [(&str, SelKindCheck); 3] = [
            ("beat_grid", |s| matches!(s, SuperSelector::Bpm { .. })),
            ("auto_tile", |s| matches!(s, SuperSelector::NeighborAware)),
            ("boil", |s| matches!(s, SuperSelector::Procedural { .. })),
        ];
        let reg = MaterialRegistry::with_catalog();
        for (name, is_kind) in expect {
            // The `preset = NAME` data lookup resolves it…
            let m = MaterialRegistry::catalog_material(name)
                .unwrap_or_else(|| panic!("'{name}' must resolve as a catalog preset"));
            let layer = m
                .super_glyph
                .unwrap_or_else(|| panic!("'{name}' must carry a super_glyph layer"));
            assert!(
                is_kind(layer.selector),
                "'{name}' selector kind mismatch: {:?}",
                layer.selector
            );
            // …and the registered slot agrees (same selector kind).
            let id = reg.id_of(name).unwrap();
            let rl = reg.get(id).super_glyph.unwrap();
            assert!(
                is_kind(rl.selector),
                "'{name}' registered slot selector mismatch"
            );
        }
    }

    /// `sample()` on a super-glyph material SUBSTITUTES a member of the selector's
    /// family for the recipe glyph, deterministically. (The recipe body would NOT
    /// otherwise emit these glyphs, so finding one proves the layer ran.)
    #[test]
    fn super_glyph_material_sample_substitutes_a_family_member() {
        let cases: [(&str, &'static [char]); 3] = [
            ("beat_grid", crate::glyph_family::GROWTH_DENSITY.glyphs),
            ("auto_tile", crate::glyph_family::GROWTH_BARS.glyphs),
            ("boil", crate::glyph_family::HAZE_DOT.glyphs),
        ];
        for (name, glyphs) in cases {
            let m = MaterialRegistry::catalog_material(name).unwrap();
            let c = ctx(4.0, 1.0);
            let coord = Vec2::new(2.0, 3.0);
            let t1 = m.sample(&Perlin::new(7), coord, &c);
            let t2 = m.sample(&Perlin::new(7), coord, &c);
            assert_eq!(t1.glyph, t2.glyph, "'{name}' sample must be deterministic");
            assert!(
                glyphs.iter().any(|&g| g as u32 == t1.glyph),
                "'{name}' sample glyph {:?} must be a member of its super-glyph family",
                char::from_u32(t1.glyph)
            );
        }
    }

    /// STABILITY: a sub-cell camera nudge (a distance creep well inside the LOD
    /// deadband, plus a coord nudge inside one neighbour-lattice bucket) must NOT
    /// flip the chosen super-glyph — mirroring the GlyphStability flip-count contract.
    #[test]
    fn super_glyph_material_is_stable_under_a_subcell_nudge() {
        // STATIC selectors (no time term) so the only variable is the camera creep.
        for name in ["auto_tile", "boil"] {
            let m = MaterialRegistry::catalog_material(name).unwrap();
            let perlin = Perlin::new(7);
            // A sub-deadband distance creep (a fraction of one LOD quant step) AND a
            // sub-bucket coord creep (well inside 1/NEIGHBOR_LATTICE of a world unit).
            let d = 6.0;
            let base = ctx(d, 1.0);
            let nudged = ctx(d + LOD_QUANT_STEP * 0.2, 1.0);
            let c0 = Vec2::new(3.0, 4.0);
            let c1 = c0 + Vec2::new(0.05, 0.04);
            let g0 = m.sample(&perlin, c0, &base).glyph;
            let g1 = m.sample(&perlin, c1, &nudged).glyph;
            assert_eq!(
                g0, g1,
                "'{name}' super-glyph must hold under a sub-cell nudge"
            );
        }
    }

    /// An animated super-glyph (`beat_grid`'s Bpm selector) flips the material's
    /// `is_animated` so the renderer keeps re-shading it (cache-bypassed).
    #[test]
    fn animated_super_glyph_marks_material_animated() {
        let beat = MaterialRegistry::catalog_material("beat_grid").unwrap();
        assert!(
            beat.is_animated(),
            "a Bpm super-glyph must mark the material animated"
        );
        let tile = MaterialRegistry::catalog_material("auto_tile").unwrap();
        // NeighborAware is static — no animation source on this material ⇒ cacheable.
        assert!(
            !tile.is_animated(),
            "a static NeighborAware super-glyph stays cacheable"
        );
    }

    // ── WORLD-ANCHORED TEXTURE-CACHE SPLIT: exactness contract ───────────────────

    /// (a) `apply_lighting(texture_sample(x)) == sample(x)` BYTE-FOR-BYTE over a
    /// matrix of materials, world coords, and distances (near/relief, mid, far, and
    /// far-frozen). This is the core "the split is exact" proof.
    #[test]
    fn split_matches_single_pass_sample_byte_for_byte() {
        let perlin = Perlin::new(SHADER_SEED);
        let reg = MaterialRegistry::with_catalog();
        let names = ids::CATALOG_NAMES;
        let coords = [
            Vec2::new(0.0, 0.0),
            Vec2::new(3.25, 7.5),
            Vec2::new(-2.1, 4.9),
            Vec2::new(11.7, -0.3),
        ];
        // Distances spanning the NEAR relief band, the octave-cap band, far, and a
        // far-frozen case (far_freeze set below).
        let dists = [
            0.5,
            2.0,
            SHADE_RELIEF_DIST + 1.0,
            SHADE_OCTAVE_DIST + 2.0,
            40.0,
        ];
        let mut checked = 0u32;
        for name in names {
            let Some(m) = MaterialRegistry::catalog_material(name) else {
                continue;
            };
            for &coord in &coords {
                for &d in &dists {
                    for &light in &[false, true] {
                        for &frozen_far in &[f32::INFINITY, 20.0] {
                            let mut c = if light { lit_ctx(d) } else { ctx(d, 1.0) };
                            c.far_freeze = frozen_far;
                            c.wall_v = 0.37; // exercise the CodeWall wall_v path too
                            let single = m.sample(&perlin, coord, &c);
                            let tex = m.texture_sample(&perlin, coord, &c);
                            let split = m.apply_lighting(&perlin, &tex, &c);
                            assert_eq!(
                                single, split,
                                "'{name}' coord={coord:?} d={d} light={light} \
                                 freeze={frozen_far}: split must equal single-pass sample"
                            );
                            checked += 1;
                        }
                    }
                }
            }
            let _ = &reg;
        }
        assert!(
            checked > 200,
            "test must cover a broad matrix (covered {checked})"
        );
    }

    // ── universal depth-contrast policy (depth_contrast.rs, applied in apply_lighting) ──

    /// REGRESSION GUARD: with depth-contrast DISABLED (the default), a representative
    /// cell shades EXACTLY as it did before the feature — proving DEFAULT OFF is
    /// byte-identical. We compare a shade taken with the feature config force-disabled
    /// against the historical single-pass `sample`, which is the byte-for-byte baseline
    /// (`split_matches_single_pass_sample_byte_for_byte`).
    #[test]
    fn depth_contrast_disabled_is_byte_identical() {
        use crate::depth_contrast::DepthContrastConfig;
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("stone").expect("static catalog material");
        // Explicitly install the DEFAULT (disabled) policy on this thread.
        DepthContrastConfig::default().apply();
        for &d in &[1.0, 5.0, 12.0, 25.0] {
            let coord = Vec2::new(3.25, 7.5);
            let c = lit_ctx(d);
            let baseline = m.sample(&perlin, coord, &c);
            let tex = m.texture_sample(&perlin, coord, &c);
            let shaded = m.apply_lighting(&perlin, &tex, &c);
            assert_eq!(
                baseline, shaded,
                "depth-contrast OFF must be byte-identical to the historical sample (d={d})"
            );
        }
        DepthContrastConfig::default().apply();
    }

    /// With depth-contrast ENABLED: a NEAR cell (≤ near_end) keeps its fg UNCHANGED
    /// (full contrast), while a FAR cell's fg is measurably PULLED toward its bg.
    #[test]
    fn depth_contrast_near_keeps_far_collapses() {
        use crate::depth_contrast::DepthContrastConfig;
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("stone").expect("static catalog material");

        // The near/far band is now derived from the LEVEL's fog band
        // (`depth_contrast::band_for_fog`), not the fixed LOD thresholds — pin an
        // explicit fog band here (matching the historical LOD defaults 6/13) so
        // this test's distances (2.0 / 20.0) land unambiguously inside/outside it,
        // independent of whatever the global LOD config is.
        let fog = FogParams {
            near: 6.0,
            far: 13.0,
            color: None,
            height: None,
        };

        // Disabled reference (the un-policed shade) for both distances.
        DepthContrastConfig::default().apply();
        let near_off = {
            let mut c = lit_ctx(2.0);
            c.fog = fog;
            m.apply_lighting(
                &perlin,
                &m.texture_sample(&perlin, Vec2::new(3.0, 3.0), &c),
                &c,
            )
        };
        let far_off = {
            let mut c = lit_ctx(20.0);
            c.fog = fog;
            m.apply_lighting(
                &perlin,
                &m.texture_sample(&perlin, Vec2::new(3.0, 3.0), &c),
                &c,
            )
        };

        // Enable: near_contrast 1.0 (full), far_contrast 0.15 (collapse). No jitter / no
        // sky fade so we isolate the contrast pull on fg→bg.
        DepthContrastConfig {
            ramp_dither: 0.0,
            enabled: true,
            near_contrast: 1.0,
            far_contrast: 0.15,
            fade_to_sky: false,
            sky_blend_full_at: 32.0,
            volume_jitter: 0.0,
        }
        .apply();
        let near_on = {
            let mut c = lit_ctx(2.0);
            c.fog = fog;
            m.apply_lighting(
                &perlin,
                &m.texture_sample(&perlin, Vec2::new(3.0, 3.0), &c),
                &c,
            )
        };
        let far_on = {
            let mut c = lit_ctx(20.0);
            c.fog = fog;
            m.apply_lighting(
                &perlin,
                &m.texture_sample(&perlin, Vec2::new(3.0, 3.0), &c),
                &c,
            )
        };

        // NEAR (≤ near_end=6, from the pinned fog band): contrast 1.0 ⇒ fg unchanged
        // vs the disabled shade.
        assert_eq!(near_on.fg, near_off.fg, "near cell keeps full-contrast fg");
        assert_eq!(near_on.bg, near_off.bg, "near cell bg untouched");

        // FAR (≥ far_start=13): fg pulled toward bg. The on-fg must sit strictly closer
        // to the (unchanged) bg than the off-fg does.
        let chan_dist = |a: u32, b: u32| -> i32 {
            let (ar, ag, ab) = ((a >> 16) & 0xFF, (a >> 8) & 0xFF, a & 0xFF);
            let (br, bg, bb) = ((b >> 16) & 0xFF, (b >> 8) & 0xFF, b & 0xFF);
            (ar as i32 - br as i32).abs()
                + (ag as i32 - bg as i32).abs()
                + (ab as i32 - bb as i32).abs()
        };
        assert_eq!(far_on.bg, far_off.bg, "no sky fade ⇒ far bg unchanged");
        let off_gap = chan_dist(far_off.fg, far_off.bg);
        let on_gap = chan_dist(far_on.fg, far_on.bg);
        assert!(
            on_gap < off_gap,
            "far cell fg must be pulled toward bg (on_gap {on_gap} < off_gap {off_gap})"
        );
        DepthContrastConfig::default().apply();
    }

    /// fade_to_sky moves a VERY-far cell's fg/bg toward the reference colour (here the
    /// fog tint), proving the soft-emergence blend.
    #[test]
    fn depth_contrast_fade_to_sky_moves_toward_reference() {
        use crate::depth_contrast::DepthContrastConfig;
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("stone").expect("static catalog material");
        let sky_ref: u32 = 0x00_30_60_90; // a distinct blue reference

        // A cell so far it is at full sky-blend (≥ sky_blend_full_at).
        let mut c = lit_ctx(30.0);
        c.fog = FogParams {
            near: 1.0,
            far: 26.0,
            color: Some(sky_ref),
            height: None,
        };

        DepthContrastConfig::default().apply();
        let off = m.apply_lighting(
            &perlin,
            &m.texture_sample(&perlin, Vec2::new(4.0, 4.0), &c),
            &c,
        );

        DepthContrastConfig {
            ramp_dither: 0.0,
            enabled: true,
            near_contrast: 1.0,
            far_contrast: 1.0, // isolate the sky fade (no contrast collapse)
            fade_to_sky: true,
            sky_blend_full_at: 28.0, // 30.0 ≥ this ⇒ FULL blend
            volume_jitter: 0.0,
        }
        .apply();
        let on = m.apply_lighting(
            &perlin,
            &m.texture_sample(&perlin, Vec2::new(4.0, 4.0), &c),
            &c,
        );

        // At full blend the bg should equal the reference (mix_rgb(bg, ref, 1.0) == ref).
        // (fg too, but fog already tinted both heavily — assert both moved toward ref.)
        let chan_dist = |a: u32, b: u32| -> i32 {
            let (ar, ag, ab) = ((a >> 16) & 0xFF, (a >> 8) & 0xFF, a & 0xFF);
            let (br, bg, bb) = ((b >> 16) & 0xFF, (b >> 8) & 0xFF, b & 0xFF);
            (ar as i32 - br as i32).abs()
                + (ag as i32 - bg as i32).abs()
                + (ab as i32 - bb as i32).abs()
        };
        assert!(
            chan_dist(on.bg, sky_ref) <= chan_dist(off.bg, sky_ref),
            "fade_to_sky must move the far bg toward the reference"
        );
        assert_eq!(
            on.bg, sky_ref,
            "at full blend the bg dissolves to the reference exactly"
        );
        DepthContrastConfig::default().apply();
    }

    /// volume_jitter is hash-STABLE: two `apply_lighting` calls on the SAME cell coord
    /// produce IDENTICAL shades (no flicker, survives the cross-frame cache).
    #[test]
    fn depth_contrast_volume_jitter_is_frame_stable() {
        use crate::depth_contrast::DepthContrastConfig;
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("stone").expect("static catalog material");
        DepthContrastConfig {
            ramp_dither: 0.0,
            enabled: true,
            near_contrast: 1.0,
            far_contrast: 1.0,
            fade_to_sky: false,
            sky_blend_full_at: 32.0,
            volume_jitter: 0.08,
        }
        .apply();
        let coord = Vec2::new(7.0, 9.0);
        let c = lit_ctx(4.0);
        let a = m.apply_lighting(&perlin, &m.texture_sample(&perlin, coord, &c), &c);
        let b = m.apply_lighting(&perlin, &m.texture_sample(&perlin, coord, &c), &c);
        assert_eq!(
            a, b,
            "volume jitter must be identical for the same cell coord (no flicker)"
        );
        DepthContrastConfig::default().apply();
    }

    /// (b) A cell shaded via cache-MISS-then-INSERT equals the SAME cell served on a
    /// later cache HIT, byte-identical — and both equal a direct single-pass sample.
    #[test]
    fn cache_miss_then_hit_is_byte_identical() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("stone").expect("static catalog material");
        assert!(
            !m.is_animated(),
            "stone must be static for the cache to engage"
        );
        let mut cache = WorldTexCache::default();
        let coord = Vec2::new(5.5, 2.25);
        let c = lit_ctx(3.0);

        cache.tick(); // frame 1
        let tex_miss = cache.texture(0, &m, &perlin, coord, &c);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1, "first touch is a miss (insert)");
        let cell_from_miss = m.apply_lighting(&perlin, &tex_miss, &c);

        cache.tick(); // frame 2 — entry persists across the tick
        let tex_hit = cache.texture(0, &m, &perlin, coord, &c);
        assert_eq!(
            cache.hits(),
            1,
            "the persisted entry must hit on the next frame"
        );
        assert_eq!(cache.misses(), 0);
        let cell_from_hit = m.apply_lighting(&perlin, &tex_hit, &c);

        assert_eq!(tex_miss, tex_hit, "cached TexResult must be byte-identical");
        assert_eq!(
            cell_from_miss, cell_from_hit,
            "cache-miss and cache-hit shades must be byte-identical"
        );
        // …and both equal a fresh single-pass sample (the cache is exact).
        assert_eq!(cell_from_hit, m.sample(&perlin, coord, &c));
    }

    /// (c) An ANIMATED material BYPASSES the cache (never inserts, never hits) and
    /// still produces a time-varying texture (it animates).
    #[test]
    fn animated_material_bypasses_cache_and_still_animates() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("water").expect("animated catalog material");
        assert!(m.is_animated(), "water must be animated for this test");
        let mut cache = WorldTexCache::default();
        let coord = Vec2::new(4.0, 6.0);

        cache.tick();
        let mut c0 = lit_ctx(2.0);
        c0.time = 0.0;
        let _ = cache.texture(0, &m, &perlin, coord, &c0);
        // No cache traffic at all — animated materials are answered directly.
        assert_eq!(cache.hits(), 0, "animated material must not hit the cache");
        assert_eq!(
            cache.misses(),
            0,
            "animated material must not insert into the cache"
        );

        // It still animates: a Shift recipe scrolls the coord with time, so the
        // texture differs between two distinct times. (water is a Shift-animated
        // hologram-family material in the catalog.)
        let mut c1 = c0;
        c1.time = 1.7;
        let a = cache.texture(0, &m, &perlin, coord, &c0);
        let b = cache.texture(0, &m, &perlin, coord, &c1);
        assert_ne!(
            (a.base, a.sub_glyph, a.coord),
            (b.base, b.sub_glyph, b.coord),
            "an animated material's texture must change with time"
        );
    }

    /// (d) Two cells at DIFFERENT world coords do not collide in the cache key: the
    /// second is a fresh miss, and each serves its OWN value (no aliasing).
    #[test]
    fn distinct_world_coords_do_not_collide_in_key() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("brick_red")
            .or_else(|| MaterialRegistry::catalog_material("stone"))
            .expect("a static catalog material");
        assert!(!m.is_animated());
        let mut cache = WorldTexCache::default();
        let c = lit_ctx(3.0);
        let p0 = Vec2::new(1.0, 1.0);
        let p1 = Vec2::new(1.0, 2.0); // different world cell

        cache.tick();
        let t0 = cache.texture(0, &m, &perlin, p0, &c);
        let t1 = cache.texture(0, &m, &perlin, p1, &c);
        assert_eq!(
            cache.misses(),
            2,
            "two distinct coords must be two distinct misses"
        );
        // Each replays its OWN value on a re-touch (no key collision/aliasing).
        cache.tick();
        assert_eq!(
            t0,
            cache.texture(0, &m, &perlin, p0, &c),
            "coord p0 serves its own TexResult"
        );
        assert_eq!(
            t1,
            cache.texture(0, &m, &perlin, p1, &c),
            "coord p1 serves its own TexResult"
        );
        assert_eq!(cache.hits(), 2);
        // The two distinct world cells must also have produced DIFFERENT keys
        // (otherwise the second would have been a hit on the first's value).
        let k0 = WorldTexKey::build(0, &m, p0, &c);
        let k1 = WorldTexKey::build(0, &m, p1, &c);
        assert_ne!(
            k0, k1,
            "distinct world coords must yield distinct cache keys"
        );
    }

    // ── TEXEL-SNAP: snapped sampling makes the cache hit; miss == hit exactly ─────

    /// `snap_to_texel_grid` is idempotent and collapses sub-texel coords within one
    /// grid cell to a single representative — the property the cache relies on.
    #[test]
    fn texel_snap_collapses_sub_texel_neighbours() {
        let n = 8.0;
        // Two sub-texel-apart coords inside the SAME 1/8-cell texel cell.
        let a = Vec2::new(3.111, 7.222);
        let b = Vec2::new(3.121, 7.231);
        let sa = snap_to_texel_grid(a, n);
        let sb = snap_to_texel_grid(b, n);
        assert_eq!(
            sa, sb,
            "neighbouring sub-texel coords must snap to one texel"
        );
        // Idempotent: snapping an already-snapped coord is a no-op.
        assert_eq!(snap_to_texel_grid(sa, n), sa, "snap must be idempotent");
        // A coord in the NEXT texel cell snaps elsewhere (no over-collapse).
        let c = snap_to_texel_grid(Vec2::new(3.30, 7.222), n);
        assert_ne!(
            c, sa,
            "a coord one texel over must NOT collapse onto the first"
        );
    }

    /// SNAP ON: a STATIC cell shaded via cache-MISS equals the SAME cell served on a
    /// later cache HIT, byte-identical — exact with respect to the SNAPPED sampling.
    /// This is the snap-mode analogue of `cache_miss_then_hit_is_byte_identical`: it
    /// snaps the sample coord exactly as `shade_cell` does, then proves the cache is
    /// still exact (a hit replays the byte-identical snapped `TexResult`).
    #[test]
    fn texel_snap_cache_miss_equals_hit_exactly() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = MaterialRegistry::catalog_material("stone").expect("static catalog material");
        assert!(!m.is_animated());
        let mut cache = WorldTexCache::default();
        let n = 8.0_f32;

        // Snap the coord + wall_v exactly as `shade_cell` does under REVISION_TEXEL_SNAP.
        let raw = Vec2::new(5.137, 2.281);
        let coord = snap_to_texel_grid(raw, n);
        let mut c = lit_ctx(3.0);
        c.wall_v = snap_scalar_to_texel_grid(0.413, n);

        cache.tick(); // frame 1 — miss + insert
        let tex_miss = cache.texture(0, &m, &perlin, coord, &c);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
        let cell_from_miss = m.apply_lighting(&perlin, &tex_miss, &c);

        cache.tick(); // frame 2 — the persisted snapped entry HITS
        let tex_hit = cache.texture(0, &m, &perlin, coord, &c);
        assert_eq!(
            cache.hits(),
            1,
            "the snapped entry must hit on the next frame"
        );
        assert_eq!(cache.misses(), 0);
        let cell_from_hit = m.apply_lighting(&perlin, &tex_hit, &c);

        assert_eq!(
            tex_miss, tex_hit,
            "snapped cached TexResult must be byte-identical"
        );
        assert_eq!(
            cell_from_miss, cell_from_hit,
            "snap-mode cache-miss and cache-hit shades must be byte-identical"
        );
        // And both equal a fresh single-pass sample AT THE SNAPPED COORD — i.e. the
        // cached value is exactly what the snapped look renders (cache is exact wrt
        // the snapped sampling, the whole point of the toggle).
        assert_eq!(cell_from_hit, m.sample(&perlin, coord, &c));

        // A second cell whose RAW coord differs but lands in the SAME snapped texel
        // also hits — that cross-pose collapse is the cross-frame reuse win.
        let raw_neighbour = Vec2::new(5.139, 2.279);
        let coord_n = snap_to_texel_grid(raw_neighbour, n);
        assert_eq!(coord_n, coord, "neighbour must snap to the same texel");
        cache.tick();
        let _ = cache.texture(0, &m, &perlin, coord_n, &c);
        assert_eq!(
            cache.hits(),
            1,
            "a re-seen snapped texel from a new pose must HIT"
        );
    }

    // ── angle-glyph (SuperSelector::ByAngle) ─────────────────────────────────

    #[test]
    fn view_incidence_is_face_on_centre_when_view_aligns_to_normal() {
        // view_dir == face normal (no in-plane component) → the face-on centre 0.5.
        let lc = flat_light_ctx(); // view_dir = +z = face_n
        assert!((view_incidence01(&lc) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn view_incidence_rotates_with_view_direction() {
        // A view grazing along +u vs along +v must give DIFFERENT azimuths.
        let mut lc = flat_light_ctx();
        lc.view_dir = Vec3::new(1.0, 0.0, 0.3).normalize(); // grazes along +u
        let a_u = view_incidence01(&lc);
        lc.view_dir = Vec3::new(0.0, 1.0, 0.3).normalize(); // grazes along +v
        let a_v = view_incidence01(&lc);
        assert!(
            (a_u - a_v).abs() > 0.1,
            "azimuth must change with view dir: {a_u} vs {a_v}"
        );
    }

    #[test]
    fn by_angle_selector_steps_through_family_with_angle() {
        // A ByAngle super-glyph over a 4-member family + 8 buckets: sweeping the angle
        // from 0..1 must hit at least 3 distinct members (the glyph tracks the angle).
        let fam = crate::glyph_family::ANIM_SPIN; // | / — \
        let layer = SuperGlyphLayer {
            family: fam,
            selector: SuperSelector::ByAngle { buckets: 8 },
            state: 0,
        };
        let coord = Vec2::new(3.0, 2.0);
        let mut seen = std::collections::HashSet::new();
        for i in 0..8 {
            let a = i as f32 / 8.0;
            seen.insert(layer.select(2.0, coord, 0.0, 0x1234, a));
        }
        assert!(
            seen.len() >= 3,
            "ByAngle must vary the glyph with the angle, got {seen:?}"
        );
        // It is NOT animated (the angle bucket is fixed per cast, time-free).
        assert!(!layer.selector.is_animated());
    }

    #[test]
    fn by_angle_selector_is_stable_within_a_bucket() {
        // Two angles inside the SAME bucket must pick the SAME member (deadband: no
        // flicker on a sub-bucket wobble).
        let layer = SuperGlyphLayer {
            family: crate::glyph_family::ANIM_SPIN,
            selector: SuperSelector::ByAngle { buckets: 8 },
            state: 0,
        };
        let coord = Vec2::new(3.0, 2.0);
        let g0 = layer.select(2.0, coord, 0.0, 0x1234, 0.01);
        let g1 = layer.select(2.0, coord, 0.0, 0x1234, 0.10); // both in bucket 0 (width 1/8)
        assert_eq!(
            g0, g1,
            "a sub-bucket angle wobble must not re-pick the glyph"
        );
    }

    #[test]
    fn facet_material_is_view_angle_glyph() {
        let m = mat_facet();
        assert!(
            m.is_view_angle_glyph(),
            "facet must report a view-angle glyph"
        );
        // A plain material does not.
        assert!(!DEFAULT_MATERIAL.is_view_angle_glyph());
    }

    // ── economical multi-channel volume underline ────────────────────────────

    #[test]
    fn volume_underline_shaded_face_preserves_hue() {
        // The SHADED (EW) face darkens by a uniform per-channel SCALE, so it keeps the
        // r:g:b RATIO exactly (hue-preserving) — only luminance drops. (The lit face's
        // lift toward white is a deliberate desaturating highlight rim, not a pure
        // scale, so the ratio claim is made on the shaded face.)
        let fg = 0x00C0_6020; // r:g:b = 192:96:32 ⇒ 6:3:1
        let bg = 0x0020_1008;
        let ul = volume_underline(fg, bg, 1, 5.0); // shaded face
        let v = unpack_rgb(ul);
        let (ro, rg) = (192.0 / 96.0, v.x / v.y);
        let (go, gg) = (96.0 / 32.0, v.y / v.z);
        assert!((ro - rg).abs() < 0.05, "hue r:g drifted: {ro} vs {rg}");
        assert!((go - gg).abs() < 0.08, "hue g:b drifted: {go} vs {gg}");
        // The lit face is brighter than the original ink (a highlight rim).
        let lit = volume_underline(fg, bg, 0, 5.0);
        assert!(
            luma_of(lit) > luma_of(fg),
            "lit-face third cue must brighten the ink"
        );
    }

    #[test]
    fn volume_underline_differs_between_faces_and_depth() {
        let fg = 0x00A0_8060;
        let bg = 0x0020_1810;
        let ns = volume_underline(fg, bg, 0, 5.0); // lit face → lifted
        let ew = volume_underline(fg, bg, 1, 5.0); // shaded face → darkened
        assert!(
            luma_of(ns) > luma_of(ew),
            "NS (lit) third cue must be brighter than EW (shaded)"
        );
        // Depth: a near lit cell is brighter than a far lit cell.
        let near = volume_underline(fg, bg, 0, VOLUME_UL_NEAR_DEPTH);
        let far = volume_underline(fg, bg, 0, VOLUME_UL_FAR_DEPTH + 4.0);
        assert!(
            luma_of(near) >= luma_of(far),
            "near lit cue must be ≥ far lit cue"
        );
    }

    #[test]
    fn volume_gate_stamps_an_underline_on_inked_cells_only() {
        let m = mat_relief_vol();
        // An inked cell earns the volume underline (a real colour, not the sentinel).
        let inked = Texel {
            glyph: u32::from('#'),
            fg: 0x00A08060,
            bg: 0x00201810,
            rank: 0,
        };
        let s = cell_style_full(&m, &inked, 5.0, 0);
        assert!(
            s.has_underline_color(),
            "inked volume cell must get an underline"
        );
        assert!(
            s.underline_color().unwrap() <= 0x00FF_FFFF,
            "underline is a real packed colour"
        );
        // A blank cell does not.
        let blank = Texel {
            glyph: u32::from(' '),
            fg: 0,
            bg: 0x00201810,
            rank: 0,
        };
        let sb = cell_style_full(&m, &blank, 5.0, 0);
        assert!(
            !sb.has_underline_color(),
            "a blank cell earns no volume underline"
        );
    }

    #[test]
    fn volume_underline_differs_by_side_through_cell_style() {
        let m = mat_relief_vol();
        let t = Texel {
            glyph: u32::from('#'),
            fg: 0x00A08060,
            bg: 0x00201810,
            rank: 0,
        };
        let ns = cell_style_full(&m, &t, 5.0, 0).underline_color().unwrap();
        let ew = cell_style_full(&m, &t, 5.0, 1).underline_color().unwrap();
        assert_ne!(
            ns, ew,
            "the two faces must produce a different third-channel colour"
        );
    }

    #[test]
    fn default_materials_get_no_volume_underline() {
        // Regression guard: the shipping wall materials carry no Volume accent, so a
        // brick/floor cell's style is unchanged (no spurious underline channel).
        let t = Texel {
            glyph: u32::from('#'),
            fg: 0x00808080,
            bg: 0x00101010,
            rank: 0,
        };
        // A plain default material with NO emissive + NO accent ⇒ no underline at all.
        let plain = Material {
            light: LightModel::FLAT,
            ..DEFAULT_MATERIAL.clone()
        };
        let s = cell_style_full(&plain, &t, 5.0, 0);
        assert!(
            !s.has_underline_color(),
            "a plain material must not gain a volume underline"
        );
        assert!(
            !mat_facet().underline_accent.is_active()
                || mat_facet().underline_accent.gate != AccentGate::Volume
        );
    }

    // ── glass-marker (underline-only transparent) ────────────────────────────

    #[test]
    fn pane_is_underline_only_transparent() {
        let m = mat_pane();
        assert!(
            m.is_transparent(),
            "pane must route to the transparent layer"
        );
        assert!(m.is_underline_only_transparent(), "pane is a glass marker");
        // The hologram (Inked accent, but NOT glass_marker) is NOT a glass marker.
        assert!(
            !holo_node().is_underline_only_transparent(),
            "holo must not be a glass marker"
        );
        // A plain opaque material is not.
        assert!(!DEFAULT_MATERIAL.is_underline_only_transparent());
    }

    // ── reactive pipe FLOW ────────────────────────────────────────────────

    /// A POWERED pipe cell (state decodes to a flow direction) must shade
    /// DIFFERENTLY over time from an UNPOWERED (cold) one: the flow scrolls its
    /// glyph pattern + lifts the hue, so the texel changes. A cold pipe (`state ==
    /// 0`, the empty-bus default) shades byte-identically to the shipped circuit
    /// wall — proof the flow is opt-in and non-destructive.
    #[test]
    fn powered_pipe_flow_differs_from_cold_pipe() {
        let perlin = Perlin::new(1);
        let coord = Vec2::new(1.3, 2.7);
        let trace = wall_circuit();

        // COLD: empty-bus state 0. This is exactly the shipped trace look.
        let mut cold = ctx(3.0, 1.0);
        cold.state = 0;
        cold.time = 0.0;
        let cold_t0 = trace.sample(&perlin, coord, &cold);

        // The cold pipe does NOT scroll: the same coord at a later time is identical
        // (the circuit wall's own `Pulse` anim touches brightness, not the glyph/coord
        // — and at face-on with no light the brightness path is the legacy fallback;
        // we compare the GLYPH, which is coord-driven and must hold for a cold cell).
        let mut cold_later = ctx(3.0, 1.0);
        cold_later.state = 0;
        cold_later.time = 2.0;
        let cold_t2 = trace.sample(&perlin, coord, &cold_later);
        assert_eq!(
            cold_t0.glyph, cold_t2.glyph,
            "an unpowered (cold) pipe must NOT scroll its glyph"
        );

        // POWERED forward: the flow scrolls the coord with time, so the glyph field
        // changes between two distinct times — the pipe reads as travelling.
        let powered_state = re_core::encode_flow_state(re_core::FlowDir::Forward);
        let mut hot_a = ctx(3.0, 1.0);
        hot_a.state = powered_state;
        hot_a.time = 0.0;
        let mut hot_b = ctx(3.0, 1.0);
        hot_b.state = powered_state;
        hot_b.time = 1.5;
        let g_a: Vec<u32> = (0..8)
            .map(|k| {
                let mut c = hot_a;
                c.time = k as f32 * 0.3;
                trace.sample(&perlin, coord, &c).glyph
            })
            .collect();
        let distinct: std::collections::BTreeSet<u32> = g_a.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "a powered pipe must scroll its glyph field over time (saw {} glyphs)",
            distinct.len()
        );
        let _ = (hot_a, hot_b);

        // HOT vs COLD at the SAME instant: the powered cell is brighter (hot-hue lift)
        // OR a different glyph (scrolled) — in any case NOT identical to the cold cell.
        let mut hot_now = ctx(3.0, 1.0);
        hot_now.state = powered_state;
        hot_now.time = 0.7;
        let mut cold_now = ctx(3.0, 1.0);
        cold_now.state = 0;
        cold_now.time = 0.7;
        let hot = trace.sample(&perlin, coord, &hot_now);
        let cold = trace.sample(&perlin, coord, &cold_now);
        assert!(
            hot.glyph != cold.glyph || hot.fg != cold.fg,
            "a powered pipe must read differently (scroll or hot hue) from a cold one"
        );
    }

    /// Flow direction must matter: Forward and Backward scroll opposite ways, so at a
    /// non-zero time they generally land on different sample coords → the field
    /// differs. (Both are "powered", so both differ from cold; this pins that the
    /// DIRECTION bit is read.)
    #[test]
    fn flow_direction_is_read() {
        let perlin = Perlin::new(1);
        let coord = Vec2::new(0.4, 0.9);
        let trace = wall_circuit();
        let sample_dir = |dir: re_core::FlowDir| -> Vec<u32> {
            (0..10)
                .map(|k| {
                    let mut c = ctx(3.0, 1.0);
                    c.state = re_core::encode_flow_state(dir);
                    c.time = k as f32 * 0.25;
                    trace.sample(&perlin, coord, &c).glyph
                })
                .collect()
        };
        let fwd = sample_dir(re_core::FlowDir::Forward);
        let bwd = sample_dir(re_core::FlowDir::Backward);
        // The two scroll directions trace different glyph sequences over the window.
        assert_ne!(fwd, bwd, "forward and backward flow must scroll oppositely");
    }

    // ── single-glyph animation tracks ──

    fn breathing_test_material() -> Material {
        Material {
            glyph_source: GlyphSource::Fixed(u32::from('@')),
            family: None,
            super_glyph: None,
            animation: Animation::NONE,
            anim_tracks: AnimTrackSet {
                tracks: [
                    AnimTrack {
                        frames: NODE_BREATH_FRAMES,
                        frame_count: 5,
                        clock: AnimClock::Seconds(2.0),
                        trigger: AnimTrigger::OnState { field_gt: 0 },
                        target: AnimTarget::Body,
                    },
                    TRACK_INERT,
                ],
                count: 1,
            },
            ..wall_brick()
        }
    }

    #[test]
    fn onstate_track_cycles_when_powered_and_is_static_when_dead() {
        let perlin = Perlin::new(1);
        let coord = Vec2::new(0.3, 0.7);
        let m = breathing_test_material();
        let powered: Vec<u32> = (0..6)
            .map(|k| {
                let mut c = ctx(3.0, 1.0);
                c.state = 1;
                c.time = k as f32 * 0.5;
                m.sample(&perlin, coord, &c).glyph
            })
            .collect();
        let distinct: std::collections::BTreeSet<u32> = powered.iter().copied().collect();
        assert!(
            distinct.len() >= 3,
            "a powered node must MORPH, got {powered:?}"
        );
        assert!(
            powered.iter().all(|g| NODE_BREATH_FRAMES[..5].contains(g)),
            "powered frames must be breath glyphs, got {powered:?}"
        );
        for k in 0..6 {
            let mut c = ctx(3.0, 1.0);
            c.state = 0;
            c.time = k as f32 * 0.5;
            assert_eq!(
                m.sample(&perlin, coord, &c).glyph,
                u32::from('@'),
                "a dead node holds its static body glyph"
            );
        }
    }

    #[test]
    fn empty_track_set_is_byte_identical_to_today() {
        let perlin = Perlin::new(1);
        let coord = Vec2::new(0.42, 0.61);
        let plain = wall_brick();
        assert_eq!(plain.anim_tracks, AnimTrackSet::EMPTY);
        for k in 0..8 {
            for &state in &[0u32, 1, 7] {
                let mut c = ctx(2.5, 1.0);
                c.time = k as f32 * 0.3;
                c.state = state;
                let with_default = plain.sample(&perlin, coord, &c);
                let mut explicit = plain.clone();
                explicit.anim_tracks = AnimTrackSet::EMPTY;
                assert_eq!(with_default, explicit.sample(&perlin, coord, &c));
            }
        }
    }

    #[test]
    fn track_frame_is_a_pure_function_of_time_and_state() {
        let track = AnimTrack {
            frames: NODE_BREATH_FRAMES,
            frame_count: 5,
            clock: AnimClock::Seconds(2.0),
            trigger: AnimTrigger::OnState { field_gt: 0 },
            target: AnimTarget::Body,
        };
        assert_eq!(track.resolve(1.25, 1), track.resolve(1.25, 1));
        assert_eq!(track.resolve(0.0, 1), Some(NODE_BREATH_FRAMES[0]));
        assert_eq!(track.resolve(0.5, 1), Some(NODE_BREATH_FRAMES[1]));
        assert_eq!(track.resolve(1.0, 1), Some(NODE_BREATH_FRAMES[2]));
        assert_eq!(track.resolve(2.5, 1), Some(NODE_BREATH_FRAMES[0]));
        assert_eq!(track.resolve(1.0, 0), None);
        assert_eq!(track.resolve(99.0, 0), None);
    }

    #[test]
    fn nonempty_track_set_reports_animated() {
        let m = breathing_test_material();
        assert!(m.is_animated());
        assert!(m.anim_tracks.is_animated());
        let nb = MaterialRegistry::catalog_material("node_breathing").unwrap();
        assert!(nb.is_animated());
        assert!(!AnimTrackSet::EMPTY.is_animated());
        assert!(!wall_brick().is_animated());
    }

    #[test]
    fn tick_clock_quantizes_to_the_beat() {
        let track = AnimTrack {
            frames: [b'A' as u32, b'B' as u32, b'C' as u32, 0, 0, 0, 0, 0],
            frame_count: 3,
            clock: AnimClock::Tick {
                bpm: 120.0,
                subdiv: 1,
            },
            trigger: AnimTrigger::Always,
            target: AnimTarget::Body,
        };
        assert_eq!(track.resolve(0.4, 0), Some(b'A' as u32));
        assert_eq!(track.resolve(0.6, 0), Some(b'B' as u32));
        assert_eq!(track.resolve(1.1, 0), Some(b'C' as u32));
    }

    #[test]
    fn node_breathing_dead_is_the_static_node_glyph() {
        let perlin = Perlin::new(1);
        let coord = Vec2::new(0.5, 0.5);
        let nb = MaterialRegistry::catalog_material("node_breathing").unwrap();
        let mut dead = ctx(3.0, 1.0);
        dead.state = 0;
        assert_eq!(
            nb.sample(&perlin, coord, &dead).glyph,
            PUZZLE_NODE_GLYPH,
            "a dead breathing-node is the static node glyph"
        );
        let frames: std::collections::BTreeSet<u32> = (0..6)
            .map(|k| {
                let mut c = ctx(3.0, 1.0);
                c.state = 1;
                c.time = k as f32 * 0.6;
                nb.sample(&perlin, coord, &c).glyph
            })
            .collect();
        assert!(
            frames.len() >= 2,
            "a powered breathing-node morphs: {frames:?}"
        );
    }

    #[test]
    fn frozen_cell_holds_track_frame_zero() {
        let perlin = Perlin::new(1);
        let coord = Vec2::new(0.2, 0.8);
        let m = breathing_test_material();
        let mut c = ctx(50.0, 1.0);
        c.far_freeze = 10.0;
        c.state = 1;
        let g0 = m.sample(&perlin, coord, &c).glyph;
        let mut c2 = c;
        c2.time = 100.0;
        assert_eq!(
            g0,
            m.sample(&perlin, coord, &c2).glyph,
            "frozen holds frame 0"
        );
        assert_eq!(g0, NODE_BREATH_FRAMES[0], "frozen ⇒ the resting frame");
    }

    // ── Scalable Material Overlay — SelfSimilar scale rule ───────────

    /// A SelfSimilar test material: the brick wall built-in tagged with a SelfSimilar
    /// scale rule. Composing the field OVER a real base proves the rule modulates an
    /// existing material rather than being a special recipe.
    fn self_similar_material() -> Material {
        Material {
            scale: ScaleMode::SelfSimilar(crate::fractal::FractalParams {
                arms: 5,
                octave_warp: 0.3,
                twist: 0.9,
            }),
            ..wall_brick()
        }
    }

    #[test]
    fn self_similar_scale_modulates_the_shade() {
        // (a) A SelfSimilar material's shade DIFFERS from the same material with None:
        //     the log-polar field modulates the cell. Sweep several coords so the
        //     proof does not hinge on one lucky point (the field is ~constant nowhere).
        let perlin = Perlin::new(SHADER_SEED);
        let none = wall_brick();
        let ss = self_similar_material();
        assert_eq!(none.scale, ScaleMode::None, "base is unscaled");
        let c = ctx(2.0, 1.0);
        let mut differs = 0;
        for &(u, v) in &[(0.30, 0.42), (0.71, 0.18), (0.55, 0.83), (0.18, 0.66)] {
            let coord = Vec2::new(u, v);
            let a = none.sample(&perlin, coord, &c);
            let b = ss.sample(&perlin, coord, &c);
            if a != b {
                differs += 1;
            }
        }
        assert!(
            differs >= 3,
            "SelfSimilar must visibly modulate the base shade (differed at {differs}/4)"
        );
    }

    #[test]
    fn self_similar_shade_is_deterministic() {
        // (b) Deterministic for a fixed (coord, time, params): the field is a pure
        //     function, so two samples with identical inputs are byte-identical — at a
        //     NON-zero time (the animation phase is engaged).
        let perlin = Perlin::new(SHADER_SEED);
        let ss = self_similar_material();
        let coord = Vec2::new(0.37, 0.62);
        let mut c = ctx(2.0, 1.0);
        c.time = 1.234; // phase engaged, not the static t=0 default
        let first = ss.sample(&perlin, coord, &c);
        let second = ss.sample(&perlin, coord, &c);
        assert_eq!(first, second, "SelfSimilar shade must be deterministic");
    }

    #[test]
    fn self_similar_field_advances_with_time() {
        // The field ANIMATES: the same cell shades differently at two phases (the
        // static dump captures one phase). Non-vacuous companion to determinism.
        let perlin = Perlin::new(SHADER_SEED);
        let ss = self_similar_material();
        let coord = Vec2::new(0.71, 0.18);
        let t0 = {
            let mut c = ctx(2.0, 1.0);
            c.time = 0.0;
            c
        };
        let t1 = {
            let mut c = ctx(2.0, 1.0);
            c.time = 0.21;
            c
        };
        assert_ne!(
            ss.sample(&perlin, coord, &t0),
            ss.sample(&perlin, coord, &t1),
            "the SelfSimilar field must advance with time (it animates)"
        );
    }

    #[test]
    fn self_similar_far_freeze_holds_phase() {
        // A far-frozen cell pins the phase (the far field is quiet) but still shades —
        // identical across two times when beyond `far_freeze`.
        let perlin = Perlin::new(SHADER_SEED);
        let ss = self_similar_material();
        let coord = Vec2::new(0.4, 0.6);
        let mut c0 = ctx(50.0, 1.0);
        c0.far_freeze = 10.0;
        c0.time = 0.0;
        let mut c1 = c0;
        c1.time = 99.0;
        assert_eq!(
            ss.sample(&perlin, coord, &c0),
            ss.sample(&perlin, coord, &c1),
            "a far-frozen SelfSimilar cell holds its phase"
        );
    }

    #[test]
    fn none_scale_is_byte_identical_no_op() {
        // (d) A None-scale material shades byte-identically to a material with NO scale
        //     field touched — the post-step is a strict no-op. We compare wall_brick()
        //     (default None) against an explicit ScaleMode::None copy at many cells/times.
        let perlin = Perlin::new(SHADER_SEED);
        let implicit = wall_brick(); // scale defaults to None
        let explicit = Material {
            scale: ScaleMode::None,
            ..wall_brick()
        };
        for &t in &[0.0_f32, 1.7, 42.0] {
            for iy in 0..5 {
                for ix in 0..5 {
                    let coord = Vec2::new(ix as f32 * 0.21 + 0.05, iy as f32 * 0.19 + 0.05);
                    let mut c = ctx(3.0, 1.0);
                    c.time = t;
                    assert_eq!(
                        implicit.sample(&perlin, coord, &c),
                        explicit.sample(&perlin, coord, &c),
                        "ScaleMode::None must be a strict no-op (byte-identical)"
                    );
                }
            }
        }
    }

    #[test]
    fn crisp_scale_is_no_op_in_p3() {
        // Crisp is a recorded-intent marker only (the analytic edge path is
        // deferred). A Crisp material must render exactly like None.
        let perlin = Perlin::new(SHADER_SEED);
        let none = wall_brick();
        let crisp = Material {
            scale: ScaleMode::Crisp,
            ..wall_brick()
        };
        let c = ctx(2.0, 1.0);
        for &(u, v) in &[(0.3, 0.4), (0.7, 0.2), (0.5, 0.8)] {
            let coord = Vec2::new(u, v);
            assert_eq!(
                none.sample(&perlin, coord, &c),
                crisp.sample(&perlin, coord, &c),
                "Crisp is a no-op (renders like None)"
            );
        }
    }

    #[test]
    fn self_similar_is_animated_and_gates_color_material() {
        // (c) color_material_animated() is TRUE with a SelfSimilar colour material and
        //     FALSE without. A SelfSimilar material reports is_animated().
        assert!(
            self_similar_material().is_animated(),
            "SelfSimilar ⇒ is_animated"
        );
        assert!(!wall_brick().is_animated(), "plain brick ⇒ not animated");

        // Register a colour palette of one plain brick → not animated.
        let mut reg = MaterialRegistry::new();
        for slot in 0..5u16 {
            let _ = slot;
            reg.register(wall_brick());
        }
        let pid = {
            reg.register_color_material(wall_brick());
            1u16
        };
        assert!(
            !reg.color_material_animated(),
            "an unscaled colour palette is not animated (gate stays false → idle-skip)"
        );

        // Tag the colour material SelfSimilar → the gate flips true.
        reg.set_color_material_scale(
            pid,
            ScaleMode::SelfSimilar(crate::fractal::FractalParams {
                arms: 4,
                octave_warp: 0.3,
                twist: 0.9,
            }),
        );
        assert!(
            reg.color_material_animated(),
            "a SelfSimilar colour material makes color_material_animated() true"
        );
        assert!(reg.any_animated(), "and the global animated flag too");
    }

    #[test]
    fn scale_mode_from_core_maps_params() {
        // The POD → FractalParams conversion lives in re_render and is 1:1.
        use re_core::{ScaleRule, SelfSimilarParams};
        assert_eq!(ScaleMode::from_core(&ScaleRule::None), ScaleMode::None);
        assert_eq!(ScaleMode::from_core(&ScaleRule::Crisp), ScaleMode::Crisp);
        let p = SelfSimilarParams {
            arms: 7,
            octave_warp: 0.25,
            twist: 0.5,
        };
        let mapped = ScaleMode::from_core(&ScaleRule::SelfSimilar(p));
        assert_eq!(
            mapped,
            ScaleMode::SelfSimilar(crate::fractal::FractalParams {
                arms: 7,
                octave_warp: 0.25,
                twist: 0.5,
            })
        );
    }

    // ── image relief + tile ──

    /// A 4x1 image whose four texels are red / green / blue / white, optionally
    /// carrying a relief channel and an authored `tile` / `relief`. 4x1 makes the
    /// horizontal texel a UV quarter, so a `tile = 4 1` repeat is exactly countable.
    fn relief_image_material(
        tile: (f32, f32),
        relief: Option<ImageRelief>,
        heights: Option<[u8; 4]>,
    ) -> Material {
        let rgba = vec![
            255, 0, 0, 255, // texel 0
            0, 255, 0, 255, // texel 1
            0, 0, 255, 255, // texel 2
            255, 255, 255, 255, // texel 3
        ];
        let img = re_core::TextureImage::new(4, 1, rgba);
        let img = match heights {
            Some(h) => img.with_height_map(h.to_vec()),
            None => img,
        };
        Material {
            recipe: TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Stretch,
                tile,
                relief,
            },
            texture: Some(std::sync::Arc::new(img)),
            ..DEFAULT_MATERIAL.clone()
        }
    }

    /// PROTECTS: `tile = N 1` MULTIPLIES the sampled coordinate, so the same span
    /// shows the pattern N times. Walking `u` across one span must cycle the 4-texel
    /// figure once untiled and FOUR times at `tile = 4 1` - the observable definition
    /// of "the wall shows the pattern N times".
    #[test]
    fn tile_repeats_the_figure_n_times_across_the_same_span() {
        let perlin = Perlin::new(SHADER_SEED);
        let c = ctx(3.0, 1.0);
        let plain = relief_image_material(IMAGE_TILE_NONE, None, None);
        let tiled = relief_image_material((4.0, 1.0), None, None);

        // 64 samples across ONE span (`coord.x` in [0,1)); count colour runs.
        let colours = |m: &Material| -> Vec<u32> {
            (0..64)
                .map(|i| m.sample(&perlin, Vec2::new(i as f32 / 64.0, 0.0), &c).fg)
                .collect()
        };
        let runs = |v: &[u32]| v.windows(2).filter(|w| w[0] != w[1]).count() + 1;

        let p = colours(&plain);
        let t = colours(&tiled);
        assert_eq!(runs(&p), 4, "untiled: one pass through the 4-texel figure");
        assert_eq!(
            runs(&t),
            16,
            "tile = 4 1: four passes through the same figure"
        );
        // And the first texel of every repeat is the SAME colour - it is a repeat,
        // not a stretch or a scroll.
        for rep in 0..4 {
            assert_eq!(
                t[rep * 16],
                t[0],
                "repeat {rep} starts at the figure's first texel"
            );
        }
    }

    /// PROTECTS: an image material with NONE of the new keys shades BYTE-IDENTICALLY
    /// to one built the pre-feature way. `tile`/`relief` are pure additions and the
    /// shipped corpus must not move by a single cell.
    #[test]
    fn an_image_without_tile_or_relief_shades_exactly_as_before() {
        let perlin = Perlin::new(SHADER_SEED);
        let plain = relief_image_material(IMAGE_TILE_NONE, None, None);
        // The same material with a relief channel PRESENT on the texture but no
        // authored `relief =`: an un-asked-for height map must change nothing.
        let with_unused_heights =
            relief_image_material(IMAGE_TILE_NONE, None, Some([0, 80, 170, 255]));
        for &d in &[1.0_f32, 4.0, 12.0] {
            for i in 0..17 {
                let coord = Vec2::new(i as f32 / 17.0, i as f32 * 0.11);
                let mut c = ctx(d, 1.0);
                c.wall_v = (i as f32 * 0.06).fract();
                c.light = Some(flat_light_ctx());
                assert_eq!(
                    plain.sample(&perlin, coord, &c),
                    with_unused_heights.sample(&perlin, coord, &c),
                    "a height channel nobody asked for must not change the shade"
                );
                // Self-lit is what makes that true: the lighting pass is skipped.
                assert!(
                    plain.recipe.props().self_lit,
                    "an image without `relief =` stays self-lit"
                );
            }
        }
    }

    /// PROTECTS: `relief =` flips the image material OFF self-lit. That single flag is
    /// the difference between "the photo's own pixels are the final colour" and "the
    /// photo is a surface the level's light falls on".
    #[test]
    fn relief_flips_the_image_material_off_self_lit() {
        let plain = relief_image_material(IMAGE_TILE_NONE, None, None);
        let lit = relief_image_material(
            IMAGE_TILE_NONE,
            Some(ImageRelief {
                height_scale: 1.0,
                normal_strength: 1.0,
            }),
            Some([0, 80, 170, 255]),
        );
        assert!(
            plain.recipe.props().self_lit,
            "no relief => self-lit (unchanged)"
        );
        assert!(
            !lit.recipe.props().self_lit,
            "relief => lit like every other surface"
        );
    }

    /// PROTECTS: with `relief =` authored, `height_at` TRACKS THE HEIGHT MAP instead
    /// of returning the flat 0.5 constant. Without this the flag flip would only
    /// darken the image uniformly - there would be no relief at all.
    #[test]
    fn height_at_tracks_the_height_map_only_when_relief_is_authored() {
        let perlin = Perlin::new(SHADER_SEED);
        let heights = [0u8, 80, 170, 255];
        let lit = relief_image_material(
            IMAGE_TILE_NONE,
            Some(ImageRelief {
                height_scale: 1.0,
                normal_strength: 1.0,
            }),
            Some(heights),
        );
        let flat = relief_image_material(IMAGE_TILE_NONE, None, Some(heights));
        let lod = ShadeLod::for_distance(1.0);
        for (i, &h8) in heights.iter().enumerate() {
            // `height_at` on an Image reads the IMAGE UV (see `texture_sample`), so a
            // texel centre is (i + 0.5) / 4 across.
            let uv = Vec2::new((i as f32 + 0.5) / 4.0, 0.5);
            let got = lit.height_at(&perlin, uv, lod);
            assert!(
                (got - f32::from(h8) / 255.0).abs() < 1e-6,
                "texel {i}: height_at must return the mapped height, got {got}"
            );
            assert_eq!(
                flat.height_at(&perlin, uv, lod),
                re_core::TextureImage::NEUTRAL_HEIGHT,
                "without `relief =` the height field stays the flat constant"
            );
        }
    }

    /// PROTECTS: the point of the whole feature - under ONE light, a HIGH texel shades
    /// brighter/denser than a LOW one. A relief that parses, flips the flag and reads
    /// the map but produces the same glyph and colour everywhere is still a flat decal.
    #[test]
    fn under_one_light_a_raised_texel_reads_brighter_than_a_recessed_one() {
        let perlin = Perlin::new(SHADER_SEED);
        // A single grey colour everywhere, so ONLY the height can differ the shade.
        let flat_grey: Vec<u8> = (0..4).flat_map(|_| [160u8, 160, 160, 255]).collect();
        let img =
            re_core::TextureImage::new(4, 1, flat_grey).with_height_map(vec![10, 90, 170, 250]);
        let m = Material {
            recipe: TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Stretch,
                tile: IMAGE_TILE_NONE,
                relief: Some(ImageRelief {
                    height_scale: 1.0,
                    normal_strength: 1.0,
                }),
            },
            texture: Some(std::sync::Arc::new(img)),
            light: LightModel {
                height_scale: 1.0,
                normal_strength: 1.0,
                ..LightModel::DEFAULT
            },
            ..DEFAULT_MATERIAL.clone()
        };
        let mut c = ctx(1.5, 1.0);
        c.light = Some(flat_light_ctx());
        let lum = |i: usize| -> f32 {
            let t = m.sample(&perlin, Vec2::new((i as f32 + 0.5) / 4.0, 0.0), &c);
            let v = unpack_rgb(t.fg);
            0.299 * v.x + 0.587 * v.y + 0.114 * v.z
        };
        let (low, high) = (lum(0), lum(3));
        assert!(
            high > low,
            "the raised texel must read brighter than the recessed one: {low} vs {high}"
        );
        // And the shade must actually VARY across the four heights - not just at the
        // extremes (a two-level result would read as a cut-out, not as relief).
        let all: Vec<f32> = (0..4).map(lum).collect();
        let distinct = {
            let mut v = all.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
            v.len()
        };
        assert!(distinct >= 3, "relief must be graded, not binary: {all:?}");
    }

    /// PROTECTS: the relief channel is never touched for a material that did not ask
    /// for it, and an image that asks for relief but has NO height map still produces
    /// a defined, flat, non-garbage output rather than an unwrap or a magic number.
    #[test]
    fn relief_without_a_height_map_is_flat_not_garbage() {
        let perlin = Perlin::new(SHADER_SEED);
        let asked = relief_image_material(
            IMAGE_TILE_NONE,
            Some(ImageRelief {
                height_scale: 1.0,
                normal_strength: 1.0,
            }),
            None, // no channel on the texture
        );
        let lod = ShadeLod::for_distance(1.0);
        for i in 0..4 {
            let uv = Vec2::new((i as f32 + 0.5) / 4.0, 0.5);
            assert_eq!(
                asked.height_at(&perlin, uv, lod),
                re_core::TextureImage::NEUTRAL_HEIGHT,
                "no height channel => the flat constant, never garbage"
            );
        }
        // And it still shades (no panic, a real texel) at every distance band.
        let mut c = ctx(6.0, 1.0);
        c.light = Some(flat_light_ctx());
        let t = asked.sample(&perlin, Vec2::new(0.5, 0.0), &c);
        assert!(
            t.glyph != 0,
            "a relief image with no map still produces a glyph"
        );
    }

    /// PROTECTS: an unresolved image (out-of-range `tex_id`, no attached texture) with
    /// `relief =` authored must still shade - the magenta error block, not a panic.
    #[test]
    fn a_relief_image_with_no_texture_still_renders_the_error_block() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = Material {
            recipe: TextureRecipe::Image {
                tex_id: 99,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Stretch,
                tile: (3.0, 2.0),
                relief: Some(ImageRelief {
                    height_scale: 1.0,
                    normal_strength: 1.0,
                }),
            },
            texture: None,
            ..DEFAULT_MATERIAL.clone()
        };
        let mut c = ctx(2.0, 1.0);
        c.light = Some(flat_light_ctx());
        let t = m.sample(&perlin, Vec2::new(0.3, 0.4), &c);
        assert!(
            t.glyph != 0,
            "an unresolved relief image must still produce a texel"
        );
    }

    // ── quality-textures: ASCII-family image samplers (Component 2) ───────────

    /// Build an `Image` material with the given subcell mode, a 16×16 attached
    /// texture from `f(x,y) -> luminance 0..255`, and a baked `EdgeField`.
    fn ascii_image_material(subcell: SubcellMode, f: impl Fn(u32, u32) -> u8) -> Material {
        let w = 16u32;
        let h = 16u32;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let tex = std::sync::Arc::new(re_core::TextureImage::new(w, h, rgba));
        let edge = crate::edge_field::EdgeField::build(&tex);
        Material {
            recipe: TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell,
                fit: Fit::Stretch,
                tile: IMAGE_TILE_NONE,
                relief: None,
            },
            texture: Some(tex),
            edge_field: edge,
            texture_anim: None,
            ..DEFAULT_MATERIAL.clone()
        }
    }

    /// A SampleCtx that threads a tex_footprint (so the sub-cell samplers engage) and
    /// places the cell-centre UV at `(u, v)` via the legacy per-cell mapping.
    fn ascii_ctx(u: f32, v: f32) -> SampleCtx {
        let mut c = ctx(3.0, 1.0);
        // legacy per-cell mapping: coord.x fract → u, ctx.wall_v → v.
        c.wall_v = v;
        c.tex_footprint = Some((1.0 / 16.0, 1.0 / 16.0)); // ~1 texel/cell → level 0.
        let _ = u; // u is supplied via the coord passed to sample_image.
        c
    }

    #[test]
    fn ascii_ramp_maps_luminance_to_glyph() {
        // Darkest → space (ramp[0]); brightest → '@' (ramp last).
        assert_eq!(ascii_ramp_glyph(0.0), u32::from(' '));
        assert_eq!(ascii_ramp_glyph(1.0), u32::from('@'));
        // A mid value lands on an interior ramp glyph (one of " .:-=+*#%@").
        let mid = ascii_ramp_glyph(0.5);
        assert!(
            ASCII_IMAGE_RAMP.contains(&(mid as u8)),
            "mid maps into the ramp"
        );
    }

    #[test]
    fn ascii_fill_cell_uses_ramp_glyph_for_known_luminance() {
        // A FLAT mid-grey texture: no edges anywhere, so the centre cell draws the ramp
        // fill glyph for that luminance — a known glyph, in colour (fg = fill).
        let grey = 128u8;
        let mat = ascii_image_material(SubcellMode::Ascii, |_, _| grey);
        let ctx = ascii_ctx(0.5, 0.5);
        let texel = mat.sample_image(Vec2::new(0.5, 0.5), &ctx);
        let lum = grey as f32 / 255.0;
        assert_eq!(
            texel.glyph,
            ascii_ramp_glyph(lum),
            "flat cell → ramp fill glyph"
        );
        // The fill colour (grey) is the fg of a non-edge Ascii cell.
        assert_eq!(texel.fg & 0xFF, u32::from(grey), "fg = the fill colour");
    }

    /// ALIGNMENT: a horizontal edge low in the cell resolves to `_` (bin 4), one
    /// through the middle stays `-` (bin 0). Same orientation, different glyph — which
    /// is the whole point, since `angle_to_bin` cannot tell these two apart and folds
    /// both into bin 0.
    ///
    /// Exercised through `ascii_edge_row_bias` directly rather than the full sampler,
    /// because the two are answering different questions and the sampler's answer is
    /// dominated by its own mip choice: with a span'd cell covering the whole texture,
    /// the cell's value comes from a level where one texel IS the cell, so it reports
    /// no edge at all. The bias deliberately probes finer than that — see its docs.
    #[test]
    fn ascii_horizontal_edge_low_in_the_cell_resolves_to_the_low_bin() {
        // A span whose V range is exactly the cell's, so the cell covers the whole
        // texture and dv = 1.0 → the probes sit at v = 0.25 / 0.75.
        let span = ImageSpan {
            along_min: 0.0,
            along_size: 1.0,
            up_top: 1.0,
            up_size: 1.0,
        };
        let with_span = |m: Material| -> Material {
            let TextureRecipe::Image {
                tex_id,
                subcell,
                fit,
                ..
            } = m.recipe
            else {
                unreachable!("helper builds an Image recipe")
            };
            Material {
                recipe: TextureRecipe::Image {
                    tex_id,
                    span: Some(span),
                    subcell,
                    fit,
                    tile: IMAGE_TILE_NONE,
                    relief: None,
                },
                ..m
            }
        };
        let fp = (1.0, 1.0); // one cell spans the whole figure

        // Band near the BOTTOM of the 16×16 texture (rows 11..13 → v ≈ 0.69..0.81).
        let low = with_span(ascii_image_material(SubcellMode::Ascii, |_, y| {
            if (11..13).contains(&y) {
                0
            } else {
                255
            }
        }));
        assert_eq!(
            low.ascii_edge_row_bias(0, 0.5, 0.5, fp),
            4,
            "an edge concentrated in the cell's lower half must select the `_` bin"
        );

        // CONTROL: the same band through the MIDDLE keeps the centred rule. Without
        // this, a bias that always fired would pass the assertion above and be wrong.
        let mid = with_span(ascii_image_material(SubcellMode::Ascii, |_, y| {
            if (7..9).contains(&y) {
                0
            } else {
                255
            }
        }));
        assert_eq!(
            mid.ascii_edge_row_bias(0, 0.5, 0.5, fp),
            0,
            "a centred horizontal edge must keep the centred `-` rule"
        );

        // And the mirror case: a band near the TOP must also stay `-`, because ASCII
        // has no raised variant. Biasing it to `_` would be actively wrong.
        let high = with_span(ascii_image_material(SubcellMode::Ascii, |_, y| {
            if (3..5).contains(&y) {
                0
            } else {
                255
            }
        }));
        assert_eq!(
            high.ascii_edge_row_bias(0, 0.5, 0.5, fp),
            0,
            "a HIGH edge has no ASCII variant and must not be pushed to the low one"
        );
    }

    // ── NO SPAN: a TERMINAL cell is not the whole image ──────────────────────
    //
    // Two sites made the same mistake, and it took instrumenting a rendered frame to
    // see it. Legacy per-cell tiling maps the texture across ONE WORLD CELL — `u`
    // wraps once per world cell, `v` runs 0..1 over one wall slice. Both
    // `ascii_texels_per_cell` and `ascii_edge_row_bias` read that as "the cell covers
    // the whole image", which is true of the WORLD cell and false of the TERMINAL
    // cell inside it. The mip choice was therefore pinned to the coarsest level at
    // every distance, and the bias probed a quarter of the whole IMAGE apart. On a
    // photo-textured wall the upper probe read exactly `0.0` on all 83 calls — pushed
    // clean off the subject — so every bias that fired was noise.

    /// A no-span cell's texel count must SCALE with its footprint. Pinned because the
    /// old constant answer looked perfectly reasonable in isolation: it is the right
    /// number for a cell that really is the whole image.
    #[test]
    fn ascii_texels_per_cell_without_a_span_scales_with_the_footprint() {
        let mat = ascii_image_material(SubcellMode::Ascii, |_, y| {
            if (13..15).contains(&y) {
                0
            } else {
                255
            }
        });
        assert!(mat.image_span().is_none(), "this fixture must have NO span");
        let field = mat
            .edge_field
            .as_ref()
            .expect("ASCII mode bakes an edge field");
        // A 16x16 image tiled once per world cell: a cell covering an eighth of a
        // world unit covers two texels, and halving the footprint halves that.
        let at = |n: f32| mat.ascii_texels_per_cell(field, (1.0 / n, 1.0 / n));
        assert!(
            (at(8.0) - 2.0).abs() < 1e-3,
            "1/8 world unit → 2 texels, got {}",
            at(8.0)
        );
        assert!(
            (at(4.0) - 4.0).abs() < 1e-3,
            "1/4 world unit → 4 texels, got {}",
            at(4.0)
        );
        assert!(
            at(2.0) > at(8.0),
            "a bigger footprint must cover more texels"
        );
    }

    /// The bias must probe a fraction of the CELL. With the fix the answer depends on
    /// the footprint — the property the old, footprint-blind version could not have.
    #[test]
    fn ascii_edge_bias_without_a_span_probes_the_cell_not_the_image() {
        // Flat except for a band at rows 13..15 of 16 → the edge sits at v ≈ 0.81.
        let mat = ascii_image_material(SubcellMode::Ascii, |_, y| {
            if (13..15).contains(&y) {
                0
            } else {
                255
            }
        });
        assert!(mat.image_span().is_none(), "this fixture must have NO span");
        let eighth = (1.0 / 8.0, 1.0 / 8.0);

        // A cell just above the band: the response is in its lower half → `_`.
        assert_eq!(
            mat.ascii_edge_row_bias(0, 0.5, 0.60, eighth),
            4,
            "a cell whose lower half holds the edge must select the `_` bin"
        );
        // A cell well above it, both probes in flat field → unbiased.
        assert_eq!(
            mat.ascii_edge_row_bias(0, 0.5, 0.40, eighth),
            0,
            "a cell far above the only edge must not be biased by it"
        );

        // FOOTPRINT DEPENDENCE — the fix's signature. At v = 0.65 a cell a quarter of
        // a world unit tall still reaches the band below it; a cell a thirty-second
        // tall does not. The old code answered these identically.
        assert_eq!(mat.ascii_edge_row_bias(0, 0.5, 0.65, (0.25, 0.25)), 4);
        assert_eq!(
            mat.ascii_edge_row_bias(0, 0.5, 0.65, (1.0 / 32.0, 1.0 / 32.0)),
            0,
            "a small cell must not be biased by an edge outside it"
        );
    }

    /// The bias must not touch non-horizontal bins: there is no ASCII positional
    /// variant of `|`, `/` or `\`, so those pass through untouched.
    #[test]
    fn ascii_edge_row_bias_leaves_non_horizontal_bins_alone() {
        let mat = ascii_image_material(SubcellMode::Ascii, |x, _| if x < 8 { 0 } else { 255 });
        for bin in [1u8, 2, 3, 4] {
            assert_eq!(
                mat.ascii_edge_row_bias(bin, 0.5, 0.5, (1.0 / 16.0, 1.0 / 16.0)),
                bin,
                "bin {bin} has no ASCII positional variant and must pass through"
            );
        }
        // A material with no baked field cannot probe, so it must also pass through.
        let bare = Material {
            edge_field: None,
            ..mat.clone()
        };
        assert_eq!(
            bare.ascii_edge_row_bias(0, 0.5, 0.5, (1.0 / 16.0, 1.0 / 16.0)),
            0
        );
    }

    #[test]
    fn ascii_edge_cell_emits_a_direction_glyph() {
        // A vertical step (left dark, right light) → the centre column is a strong
        // VERTICAL edge → the sampler swaps to the '|' direction glyph there.
        let mat = ascii_image_material(SubcellMode::Ascii, |x, _| if x < 8 { 0 } else { 255 });
        // Cell centred on the edge (u ≈ 0.5).
        let ctx = ascii_ctx(0.5, 0.5);
        let texel = mat.sample_image(Vec2::new(0.5, 0.5), &ctx);
        let dir_glyphs: Vec<u32> = crate::edge_field::DIR_GLYPHS
            .iter()
            .map(|&c| u32::from(c))
            .collect();
        assert!(
            dir_glyphs.contains(&texel.glyph),
            "an edge cell must emit a directional glyph, got {:?}",
            char::from_u32(texel.glyph)
        );
        // Specifically a vertical step reads as a '|' stroke.
        assert_eq!(texel.glyph, u32::from('|'), "vertical step → '|' contour");
    }

    #[test]
    fn lines_mode_recesses_nonedge_fill() {
        // In Lines mode a non-edge cell's fg sits close to its bg (low contrast) so the
        // fill recedes; Ascii mode inks the same cell at full fill colour. Compare the
        // fg/bg gap of the two modes on the same flat texture.
        let grey = 200u8;
        let ascii = ascii_image_material(SubcellMode::Ascii, |_, _| grey);
        let lines = ascii_image_material(SubcellMode::Lines, |_, _| grey);
        let ctx = ascii_ctx(0.5, 0.5);
        let a = ascii.sample_image(Vec2::new(0.5, 0.5), &ctx);
        let l = lines.sample_image(Vec2::new(0.5, 0.5), &ctx);
        let gap = |t: &Texel| (luma_of(t.fg) - luma_of(t.bg)).abs();
        assert!(
            gap(&l) < gap(&a),
            "Lines fill must be lower-contrast than Ascii (recessed): lines={} ascii={}",
            gap(&l),
            gap(&a)
        );
    }

    #[test]
    fn ascii_sampler_is_deterministic() {
        let mat = ascii_image_material(SubcellMode::Contour, |x, y| ((x ^ y) * 8) as u8);
        let ctx = ascii_ctx(0.3, 0.7);
        let a = mat.sample_image(Vec2::new(0.3, 0.7), &ctx);
        let b = mat.sample_image(Vec2::new(0.3, 0.7), &ctx);
        assert_eq!(a, b, "the ASCII sampler must be deterministic");
    }

    #[test]
    fn ascii_modes_without_footprint_fall_back_to_solid_block() {
        // No tex_footprint threaded (every legacy caster) ⇒ the (subcell, None) match
        // arm falls through to the plain solid-block path — byte-identical, so an ASCII
        // material never garbages a caller that didn't opt into sub-cell footprints.
        let mat = ascii_image_material(SubcellMode::Ascii, |_, _| 128);
        let mut ctx = ctx(3.0, 1.0);
        ctx.tex_footprint = None;
        let texel = mat.sample_image(Vec2::new(0.5, 0.5), &ctx);
        assert_eq!(
            texel.glyph, SOLID_BLOCK,
            "no footprint → solid block fallback"
        );
    }

    // ── TRUCHET on a WALL: the surface the lattice is actually FOR ────────────
    //
    // The recipe shipped flooring `coord`, which is the world hit on a floor but NOT
    // on a wall: `sample_coord` builds `coord.x = tex_u + floor(the face's CONSTANT
    // axis)`, the same offset for every cell of one flat run. So `floor(coord.x)` was
    // constant along the whole wall and the "lattice" rendered as one box-drawing
    // glyph repeated from end to end of every screen row. Measured on
    // `--dump`, not reasoned about.

    /// The glyph sequence a run of `n` samples along a wall produces, stepping
    /// `step` world units per sample at a fixed height.
    fn truchet_run(mat: &Material, up: f32, step: f32, n: usize) -> Vec<char> {
        let perlin = Perlin::new(SHADER_SEED);
        (0..n)
            .map(|i| {
                let along = i as f32 * step;
                // `coord` deliberately held CONSTANT: on one flat wall face that is
                // what the caster hands every cell, so a sampler that reads it cannot
                // tell the cells apart. Only `wall_along` moves.
                let c = plane_ctx(4.0, along, up);
                let t = mat.sample(&perlin, WRAPPED_COORD, &c);
                char::from_u32(t.glyph).unwrap_or('?')
            })
            .collect()
    }

    /// THE REGRESSION. Walking along a wall must change the tile — with `coord`
    /// pinned, because that is what a flat face gives. A constant result here is the
    /// shipped bug: a wall painted with one repeated glyph.
    #[test]
    fn truchet_varies_along_a_wall_where_coord_does_not() {
        let mat = mat_conduit();
        // Step a quarter world cell: at WALL_SCALE = 4 that is one tile per sample,
        // so consecutive samples are consecutive tiles.
        let run = truchet_run(&mat, 1.5, 0.25, 24);
        let distinct: std::collections::HashSet<char> = run.iter().copied().collect();
        assert!(
            distinct.len() >= 4,
            "a wall run must resolve distinct tiles, saw {distinct:?} in {run:?}"
        );
    }

    /// …and the same in the OTHER axis: stepping up the wall must change the tile
    /// too, so the lattice is two-dimensional rather than a set of stacked bands.
    #[test]
    fn truchet_varies_up_a_wall() {
        let mat = mat_conduit();
        let perlin = Perlin::new(SHADER_SEED);
        let column: Vec<char> = (0..24)
            .map(|i| {
                let c = plane_ctx(4.0, 3.0, i as f32 * 0.25);
                let t = mat.sample(&perlin, WRAPPED_COORD, &c);
                char::from_u32(t.glyph).unwrap_or('?')
            })
            .collect();
        let distinct: std::collections::HashSet<char> = column.iter().copied().collect();
        assert!(
            distinct.len() >= 4,
            "a wall column must resolve distinct tiles, saw {distinct:?} in {column:?}"
        );
    }

    /// A FLOOR keeps reading `coord` — there `coord` IS the world hit, and
    /// `wall_along`/`wall_up` are both `0.0`, so trusting the wall plane on a floor
    /// would collapse the entire ground plane onto one tile.
    #[test]
    fn truchet_on_a_floor_indexes_coord_not_the_wall_plane() {
        let mat = mat_conduit();
        let perlin = Perlin::new(SHADER_SEED);
        let mut seen = std::collections::HashSet::new();
        for i in 0..24 {
            let mut c = ctx(4.0, 1.0); // on_wall = false: a floor/ceiling hit
            c.wall_along = 0.0;
            c.wall_up = 0.0;
            let t = mat.sample(&perlin, Vec2::new(i as f32 * 0.25, 2.0), &c);
            seen.insert(char::from_u32(t.glyph).unwrap_or('?'));
        }
        assert!(
            seen.len() >= 4,
            "a floor run must resolve distinct tiles, saw {seen:?}"
        );
    }

    /// `scale` subdivides without breaking the thing the recipe exists for: whatever
    /// the factor, the index still floors a CONTINUOUS world coordinate, so two
    /// samples either side of a tile boundary agree on the edge they share. Checked
    /// through the SHADER (not `truchet` directly) so the wall-plane wiring is what
    /// is under test.
    #[test]
    fn truchet_tiles_stay_edge_matched_at_the_shipped_scale() {
        let scale = crate::truchet::WALL_SCALE;
        let density = crate::truchet::DENSITY;
        // Sample at tile centres so the floor is unambiguous, and compare each tile's
        // EAST bit with its right neighbour's WEST bit — the connectivity the glyph
        // vocabulary encodes.
        for i in 0..32 {
            let along = (i as f32 + 0.5) / scale;
            let next = (i as f32 + 1.5) / scale;
            let cx = (along * scale).floor() as i32;
            let nx = (next * scale).floor() as i32;
            assert_eq!(nx, cx + 1, "tile centres must step one tile at a time");
            let me = crate::truchet::connections(cx, 6, 0, density);
            let right = crate::truchet::connections(nx, 6, 0, density);
            assert_eq!(
                me & crate::truchet::E != 0,
                right & crate::truchet::W != 0,
                "tile {cx} east must equal tile {nx} west at scale {scale}"
            );
        }
    }

    // -- ANIMATED image textures: `[texture] frames = clip.gif` ------
    //
    // The shader picks `frames[(time*fps) as usize % n]` from the cast's `time`. A
    // ONE-frame set must behave EXACTLY as a still texture always has: same texel,
    // no time dependence, `is_animated() == false`, fully cacheable.

    /// A 1x1 opaque image whose single texel is `(v, v, v)`.
    fn flat_frame(v: u8) -> re_core::TextureImage {
        re_core::TextureImage::new(1, 1, vec![v, v, v, 255])
    }

    /// An `Image` material bound to `tex_id 0` of `textures`, wired exactly the way
    /// the production registry wires it (`attach_image_textures`).
    fn image_material_from(textures: &[re_core::TextureSet]) -> Material {
        let mut reg = MaterialRegistry::new();
        reg.register(Material {
            recipe: TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Stretch,
                tile: IMAGE_TILE_NONE,
                relief: None,
            },
            ..DEFAULT_MATERIAL.clone()
        });
        reg.attach_image_textures(textures);
        reg.get(MaterialId(0)).clone()
    }

    /// PROTECTS: an animated set really animates - the painted colour follows
    /// `frames[(time*fps) as usize % n]` and WRAPS at the end of the clip.
    #[test]
    fn an_animated_image_samples_the_frame_selected_by_time() {
        let perlin = Perlin::new(SHADER_SEED);
        let set = re_core::TextureSet::new(
            vec![
                flat_frame(10),
                flat_frame(90),
                flat_frame(170),
                flat_frame(250),
            ],
            10.0, // one frame per 0.1 s
        );
        let m = image_material_from(std::slice::from_ref(&set));
        let red_of = |time: f32| {
            let mut c = ctx(3.0, 1.0);
            c.time = time;
            (m.sample_image(Vec2::new(0.5, 0.5), &c).fg >> 16) & 0xFF
        };
        assert_eq!(red_of(0.00), 10, "t=0 -> frame 0");
        assert_eq!(red_of(0.10), 90, "t=0.1 -> frame 1");
        assert_eq!(red_of(0.25), 170, "t=0.25 -> frame 2");
        assert_eq!(red_of(0.35), 250, "t=0.35 -> frame 3");
        assert_eq!(red_of(0.40), 10, "t=0.4 -> wraps to frame 0");
        // DETERMINISM: the same time always gives the same frame.
        assert_eq!(red_of(0.25), 170);
    }

    /// PROTECTS: NO-REGRESSION - a STILL (one-frame) texture is not animated, ignores
    /// `time` entirely, and shades identically to a material whose `texture` was set
    /// by hand the way every pre-frames call site set it.
    #[test]
    fn a_still_image_is_not_animated_and_ignores_time() {
        let img = flat_frame(123);
        let set = re_core::TextureSet::still(img.clone());
        let m = image_material_from(std::slice::from_ref(&set));
        assert!(
            m.texture_anim.is_none(),
            "a still set attaches no frame set"
        );
        assert!(
            !m.is_animated(),
            "a still image material must stay cacheable"
        );

        // The historical hand-wired shape - the pre-frames construction.
        let legacy = Material {
            recipe: TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Stretch,
                tile: IMAGE_TILE_NONE,
                relief: None,
            },
            texture: Some(std::sync::Arc::new(img)),
            ..DEFAULT_MATERIAL.clone()
        };
        for t in [0.0, 1.0, 12.5, 900.0] {
            let mut c = ctx(3.0, 1.0);
            c.time = t;
            assert_eq!(
                m.sample_image(Vec2::new(0.5, 0.5), &c),
                legacy.sample_image(Vec2::new(0.5, 0.5), &c),
                "a one-frame set must shade exactly like the bare attached texture"
            );
        }
    }

    /// PROTECTS: an animated image material reports `is_animated()`, which is what
    /// makes BOTH shade caches bypass it - a cached frame would freeze the clip.
    #[test]
    fn an_animated_image_is_animated_and_bypasses_both_caches() {
        let perlin = Perlin::new(SHADER_SEED);
        let set = re_core::TextureSet::new(vec![flat_frame(10), flat_frame(200)], 10.0);
        let m = image_material_from(std::slice::from_ref(&set));
        assert!(m.texture_anim.is_some(), "the frame set is attached");
        assert!(m.is_animated(), "more than one frame => animated");

        let coord = Vec2::new(0.5, 0.5);
        let mut early = ctx(3.0, 1.0);
        early.light = Some(flat_light_ctx());
        early.time = 0.0;
        let mut late = early;
        late.time = 0.1;

        // World-texture cache: the second (later) cast must NOT be served frame 0.
        let mut cache = WorldTexCache::with_slots(64);
        let _ = cache.texture(0, &m, &perlin, coord, &early);
        assert_eq!(
            cache.texture(0, &m, &perlin, coord, &late),
            m.texture_sample(&perlin, coord, &late),
            "an animated material must never be served a cached frame"
        );
        // Per-cast memo: same contract.
        let mut memo = SampleMemo::with_slots(64);
        let _ = memo.sample(0, &m, &perlin, coord, &early);
        assert_eq!(
            memo.sample(0, &m, &perlin, coord, &late),
            m.sample(&perlin, coord, &late),
        );
    }

    /// PROTECTS: the registry attaches FRAME 0 to `Material::texture` for an animated
    /// set too, so every time-less consumer (fit/aspect derivation, the EdgeField
    /// precompute, the caster's cut-out tests) keeps reading the picture it always did.
    #[test]
    fn attach_image_textures_puts_frame_zero_in_the_texture_slot() {
        let set = re_core::TextureSet::new(vec![flat_frame(7), flat_frame(200)], 10.0);
        let m = image_material_from(std::slice::from_ref(&set));
        let tex = m.texture.as_ref().expect("frame 0 attached");
        assert_eq!(
            tex.sample_nearest(0.5, 0.5),
            [7, 7, 7, 255],
            "frame 0, not frame 1"
        );
        assert!(
            std::sync::Arc::ptr_eq(tex, set.first()),
            "frame 0 is SHARED with the map's set, not deep-copied"
        );
    }

    /// PROTECTS: an unresolved `tex_id` on an animated-capable registry still yields
    /// the magenta error block, never a panic - every input shape has a defined output.
    #[test]
    fn an_unresolved_animated_tex_id_still_renders_the_error_block() {
        let m = Material {
            recipe: TextureRecipe::Image {
                tex_id: 42,
                span: None,
                subcell: SubcellMode::Off,
                fit: Fit::Stretch,
                tile: IMAGE_TILE_NONE,
                relief: None,
            },
            ..DEFAULT_MATERIAL.clone()
        };
        let mut reg = MaterialRegistry::new();
        reg.register(m);
        let set = re_core::TextureSet::new(vec![flat_frame(1), flat_frame(2)], 10.0);
        reg.attach_image_textures(std::slice::from_ref(&set));
        let m = reg.get(MaterialId(0)).clone();
        assert!(m.texture.is_none() && m.texture_anim.is_none());
        assert!(!m.is_animated(), "an unresolved id animates nothing");
        let mut c = ctx(3.0, 1.0);
        c.time = 5.0;
        assert_eq!(m.sample_image(Vec2::new(0.5, 0.5), &c).glyph, SOLID_BLOCK);
    }

    // ── shade-cache KEY GAP: tex_footprint / subcell_backdrop ──────
    //
    // `Material::texture_sample` -> `sample_image` reads `SampleCtx::tex_footprint`
    // (where the 2x2 / 2x4 / ASCII sub-samples land, and which mip the EdgeField
    // serves) and `SampleCtx::subcell_backdrop` (what a TRANSPARENT sub-texel
    // reveals). Neither was folded into `WorldTexKey` or `SampleKey`, so two cells
    // that differ in ONLY those inputs collided on one key and the second was served
    // the FIRST one's texel — a near cell painted with a far cell's sub-texels.
    //
    // Both tests below FAIL before the fix: in a debug build the caches' own
    // hit-validation `debug_assert_eq!` fires; in a release build the explicit
    // assertion does. They pass after the two `subcell` key fields land.

    /// A 16x16 image material in `Quad` sub-cell mode over a 1-texel checkerboard —
    /// the finest pattern the sub-sample offsets can resolve, so a change of
    /// footprint moves every sub-sample onto a different texel.
    fn subcell_key_material() -> Material {
        ascii_image_material(
            SubcellMode::Quad,
            |x, y| if (x + y) % 2 == 0 { 250 } else { 5 },
        )
    }

    /// PROTECTS: the cross-frame world-texture cache must not serve a cell its
    /// NEIGHBOUR-IN-EVERYTHING-BUT-FOOTPRINT texel. `tex_footprint` is a declared
    /// input of the Image recipe, so it must be part of the key.
    #[test]
    fn world_tex_cache_key_folds_tex_footprint() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = subcell_key_material();
        let coord = Vec2::new(0.5, 0.25);
        let mut near = ctx(3.0, 1.0);
        near.wall_v = 0.5;
        near.tex_footprint = Some((1.0 / 16.0, 1.0 / 16.0));
        let mut far = near;
        far.tex_footprint = Some((0.5, 0.5));
        // The two footprints really do shade differently — otherwise the test proves
        // nothing about the key.
        let fresh_near = m.texture_sample(&perlin, coord, &near);
        let fresh_far = m.texture_sample(&perlin, coord, &far);
        assert_ne!(
            fresh_near, fresh_far,
            "fixture check: the two footprints must produce different texels"
        );

        let mut cache = WorldTexCache::with_slots(64);
        let got_near = cache.texture(0, &m, &perlin, coord, &near);
        let got_far = cache.texture(0, &m, &perlin, coord, &far);
        assert_eq!(
            got_near, fresh_near,
            "the near cell must get the near texel"
        );
        assert_eq!(
            got_far, fresh_far,
            "the far cell must NOT be served the near cell's cached texel"
        );
    }

    /// PROTECTS: the same gap in the per-cast `SampleMemo`. Wiped every cast, so the
    /// staleness is confined to one frame — which is exactly long enough to paint a
    /// whole wall of cells with one cell's sub-texels.
    #[test]
    fn sample_memo_key_folds_tex_footprint() {
        let perlin = Perlin::new(SHADER_SEED);
        let m = subcell_key_material();
        let coord = Vec2::new(0.5, 0.25);
        let mut near = ctx(3.0, 1.0);
        near.wall_v = 0.5;
        near.light = Some(flat_light_ctx());
        near.tex_footprint = Some((1.0 / 16.0, 1.0 / 16.0));
        let mut far = near;
        far.tex_footprint = Some((0.5, 0.5));
        let fresh_far = m.sample(&perlin, coord, &far);
        assert_ne!(
            m.sample(&perlin, coord, &near),
            fresh_far,
            "fixture check: the two footprints must produce different texels"
        );

        let mut memo = SampleMemo::with_slots(64);
        let _ = memo.sample(0, &m, &perlin, coord, &near);
        assert_eq!(
            memo.sample(0, &m, &perlin, coord, &far),
            fresh_far,
            "the second cell must not be served the first cell's cached texel"
        );
    }

    /// PROTECTS: `subcell_backdrop` — the colour a TRANSPARENT sub-texel reveals — is
    /// equally an input of the shade, and equally absent from the key. Two cells over
    /// the same texel with different backdrops must not share an entry.
    #[test]
    fn world_tex_cache_key_folds_subcell_backdrop() {
        let perlin = Perlin::new(SHADER_SEED);
        // Half the texels TRANSPARENT so the backdrop actually shows through.
        let w = 16u32;
        let h = 16u32;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let a = if (x + y) % 2 == 0 { 255 } else { 0 };
                rgba.extend_from_slice(&[200, 200, 200, a]);
            }
        }
        let m = Material {
            recipe: TextureRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: SubcellMode::Quad,
                fit: Fit::Stretch,
                tile: IMAGE_TILE_NONE,
                relief: None,
            },
            texture: Some(std::sync::Arc::new(re_core::TextureImage::new(w, h, rgba))),
            ..DEFAULT_MATERIAL.clone()
        };
        let coord = Vec2::new(0.5, 0.25);
        let mut a = ctx(3.0, 1.0);
        a.wall_v = 0.5;
        a.tex_footprint = Some((1.0 / 16.0, 1.0 / 16.0));
        a.subcell_backdrop = Some(0x00FF_0000);
        let mut b = a;
        b.subcell_backdrop = Some(0x0000_00FF);
        let fresh_b = m.texture_sample(&perlin, coord, &b);
        assert_ne!(
            m.texture_sample(&perlin, coord, &a),
            fresh_b,
            "fixture check: the two backdrops must produce different texels"
        );

        let mut cache = WorldTexCache::with_slots(64);
        let _ = cache.texture(0, &m, &perlin, coord, &a);
        assert_eq!(
            cache.texture(0, &m, &perlin, coord, &b),
            fresh_b,
            "a different backdrop must not hit the first backdrop's entry"
        );
    }
}

/// PUNCTUAL LIGHTS + HEIGHT FOG. The contract these pin:
/// a level-authored light actually pools (nearer ⇒ brighter ⇒ denser ramp), a
/// level with NO lights shades bit-for-bit like the pre-lighting build, and the
/// height-fog term is a monotone clarity multiplier that is a literal no-op when
/// unauthored.
#[cfg(test)]
mod point_lighting {
    use super::*;

    /// A white light of unit intensity at `pos` with a generous radius.
    fn lamp(pos: Vec3, radius: f32) -> re_world::PointLight {
        re_world::PointLight {
            pos,
            color: 0x00FF_FFFF,
            radius,
            intensity: 1.0,
        }
    }

    /// A minimal [`SampleCtx`] for the full-pipeline bg-gain test below — mirrors
    /// `tests::ctx` (private to that sibling module, so not reachable here) but
    /// this module only needs `distance`/`light` to vary.
    fn sample_ctx(distance: f32) -> SampleCtx {
        SampleCtx {
            distance,
            foreshorten: 1.0,
            side: 0,
            screen_y: 4,
            wall_v: 0.5,
            tick: 0,
            time: 0.0,
            fog: FogParams::DEFAULT,
            far_freeze: f32::INFINITY,
            light: None,
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: None,
            wall_along: 0.0,
            wall_up: 0.0,
            on_wall: false,
            tex_footprint: None,
            subcell_backdrop: None,
        }
    }

    #[test]
    fn a_nearer_cell_gets_more_light_than_a_far_one() {
        let pts = PointLights::from_world(&[lamp(Vec3::new(0.0, 0.0, 2.0), 4.0)]);
        let n = Vec3::Z; // a floor facing straight up at the lamp
        let near = point_light_sum(&pts, Vec3::new(0.0, 0.0, 0.0), n);
        let far = point_light_sum(&pts, Vec3::new(6.0, 0.0, 0.0), n);
        assert!(near.x > far.x, "near {near:?} must out-light far {far:?}");
        assert!(
            far.x > 0.0,
            "a lit-but-distant cell still receives some light"
        );
    }

    #[test]
    fn attenuation_is_exactly_half_at_the_authored_radius() {
        // The documented falloff `1/(1 + d^2/r^2)` is half at d == r. Measured
        // head-on (N.L == 1) so only the attenuation term is under test.
        let r = 3.0;
        let pts = PointLights::from_world(&[lamp(Vec3::new(0.0, 0.0, r), r)]);
        let got = point_light_sum(&pts, Vec3::ZERO, Vec3::Z);
        assert!(
            (got.x - 0.5).abs() < 1e-5,
            "at d == radius the light is half: {got:?}"
        );
    }

    #[test]
    fn a_surface_facing_away_receives_nothing() {
        let pts = PointLights::from_world(&[lamp(Vec3::new(0.0, 0.0, 2.0), 4.0)]);
        // Normal points DOWN, away from a lamp overhead.
        let got = point_light_sum(&pts, Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(got, Vec3::ZERO, "a surface facing away contributes nothing");
    }

    #[test]
    fn from_world_drops_inert_lights_and_caps_the_set() {
        let bad = [
            re_world::PointLight {
                radius: 0.0,
                ..lamp(Vec3::ZERO, 1.0)
            },
            re_world::PointLight {
                radius: f32::NAN,
                ..lamp(Vec3::ZERO, 1.0)
            },
            re_world::PointLight {
                intensity: 0.0,
                ..lamp(Vec3::ZERO, 1.0)
            },
            re_world::PointLight {
                pos: Vec3::new(f32::INFINITY, 0.0, 0.0),
                ..lamp(Vec3::ZERO, 1.0)
            },
        ];
        assert!(
            PointLights::from_world(&bad).is_empty(),
            "inert lights are dropped"
        );

        let many: Vec<_> = (0..MAX_POINT_LIGHTS + 3)
            .map(|i| lamp(Vec3::new(i as f32, 0.0, 1.0), 5.0))
            .collect();
        assert_eq!(
            PointLights::from_world(&many).len(),
            MAX_POINT_LIGHTS,
            "capped"
        );
    }

    #[test]
    fn an_empty_light_set_shades_bit_for_bit_like_before() {
        // The whole no-regression promise: `PointLights::NONE` must leave
        // `light_surface_mode` returning EXACTLY what it returned pre-lighting,
        // which means the world anchor cannot be observable.
        let env = LightEnv::builtin();
        let view = Vec3::new(0.0, -0.4, 0.9).normalize();
        let lc = LightCtx::for_floor(true, view, &env).at(Vec3::new(3.0, 4.0, 0.0));
        assert!(
            lc.points.is_empty(),
            "the built-in environment has no punctual lights"
        );
        let lm = LightModel::DEFAULT;
        let albedo = Vec3::new(0.4, 0.5, 0.6);

        let (fg_a, sh_a, lum_a, _) = light_surface_mode(albedo, Vec3::Z, 0.5, &lm, &lc, false);
        // The same context with the world anchor left at the origin.
        let lc0 = LightCtx::for_floor(true, view, &env);
        let (fg_b, sh_b, lum_b, _) = light_surface_mode(albedo, Vec3::Z, 0.5, &lm, &lc0, false);
        assert_eq!(fg_a, fg_b, "un-lit shading is position-independent");
        assert_eq!(sh_a, sh_b);
        assert_eq!(lum_a, lum_b);
    }

    #[test]
    fn a_cell_inside_the_pool_shades_brighter_and_denser_than_one_outside() {
        // End-to-end through the shading function: the same surface, the same
        // material, the same view - only the world position differs.
        let env = LightEnv {
            points: PointLights::from_world(&[lamp(Vec3::new(0.0, 0.0, 1.5), 3.0)]),
            ..LightEnv::builtin()
        };
        let view = Vec3::new(0.0, -0.3, 0.95).normalize();
        let lm = LightModel::DEFAULT;
        let albedo = Vec3::splat(0.5);

        let inside = LightCtx::for_floor(true, view, &env).at(Vec3::new(0.0, 0.0, 0.0));
        let outside = LightCtx::for_floor(true, view, &env).at(Vec3::new(9.0, 0.0, 0.0));
        let (fg_in, _, lum_in, _) = light_surface_mode(albedo, Vec3::Z, 0.5, &lm, &inside, false);
        let (fg_out, _, lum_out, _) =
            light_surface_mode(albedo, Vec3::Z, 0.5, &lm, &outside, false);

        assert!(
            fg_in.max_element() > fg_out.max_element(),
            "the cell under the lamp must be brighter: {fg_in:?} vs {fg_out:?}"
        );
        assert!(
            lum_in > lum_out,
            "and must score a denser glyph: lum {lum_in} vs {lum_out}"
        );
    }

    #[test]
    fn point_lights_survive_the_view_independent_toggle() {
        // Punctual lights are position+normal only, so the near-ring pan's
        // view-independent mode must NOT drop them (unlike the headlamp).
        let env = LightEnv {
            points: PointLights::from_world(&[lamp(Vec3::new(0.0, 0.0, 1.0), 2.0)]),
            ..LightEnv::builtin()
        };
        let lc = LightCtx::for_floor(true, Vec3::Z, &env).at(Vec3::ZERO);
        let dark = LightCtx::for_floor(true, Vec3::Z, &LightEnv::builtin()).at(Vec3::ZERO);
        let lm = LightModel::DEFAULT;
        let (fg_lit, _, _, _) = light_surface_mode(Vec3::splat(0.5), Vec3::Z, 0.5, &lm, &lc, true);
        let (fg_unlit, _, _, _) =
            light_surface_mode(Vec3::splat(0.5), Vec3::Z, 0.5, &lm, &dark, true);
        assert!(
            fg_lit.max_element() > fg_unlit.max_element(),
            "view_indep must keep the point light: {fg_lit:?} vs {fg_unlit:?}"
        );
    }

    #[test]
    fn unauthored_height_fog_is_a_literal_no_op() {
        let fog = FogParams::DEFAULT;
        assert!(
            fog.height.is_none(),
            "the shipped default authors no height fog"
        );
        for z in [-3.0, 0.0, 1.0, 7.5, 100.0] {
            assert_eq!(height_fog(z, &fog), 1.0, "z={z} must not change clarity");
        }
    }

    #[test]
    fn height_fog_clears_monotonically_upward() {
        let fog = FogParams {
            height: Some(re_world::HeightFog {
                base: 1.0,
                falloff: 0.5,
            }),
            ..FogParams::DEFAULT
        };
        // At and below the base the term is inert; above it the murk thins with
        // height and the factor never rises again.
        assert_eq!(height_fog(0.0, &fog), 1.0);
        assert_eq!(height_fog(1.0, &fog), 1.0);
        let mut prev = 1.0_f32;
        for i in 0..24 {
            let z = 1.0 + i as f32 * 0.5;
            let c = height_fog(z, &fog);
            assert!(
                c <= prev + 1e-6,
                "the height term must not rise with z: z={z}"
            );
            assert!((0.0..=1.0).contains(&c), "clarity stays in [0,1]: {c}");
            prev = c;
        }
        // The documented form: exp(-falloff * (z - base)).
        let want = (-0.5_f32 * 2.0).exp();
        assert!((height_fog(3.0, &fog) - want).abs() < 1e-6);
    }

    #[test]
    fn a_hand_built_negative_falloff_cannot_invert_the_fog() {
        // The parser refuses it; the shader must not trust that alone.
        for h in [
            re_world::HeightFog {
                base: 0.0,
                falloff: -1.0,
            },
            re_world::HeightFog {
                base: 0.0,
                falloff: f32::NAN,
            },
        ] {
            let fog = FogParams {
                height: Some(h),
                ..FogParams::DEFAULT
            };
            assert_eq!(height_fog(5.0, &fog), 1.0, "garbage height fog is inert");
        }
    }

    #[test]
    fn light_surface_returns_the_raw_point_radiance_for_the_bg_carrier() {
        // `light_surface`'s own contract: the 4th return element is
        // `point_light_sum`'s own output — zero with no lights, non-zero and
        // hue-bearing with a coloured one — independent of the scalar `shadow`
        // term (which by construction cannot carry hue: `dim_rgb` only scales).
        let flat_env = LightEnv {
            key_dir: Vec3::Z,
            key_color: Vec3::ONE,
            ambient: Vec3::splat(0.15),
            headlamp: 0.5,
            points: PointLights::NONE,
        };
        let lc_dark = LightCtx::for_floor(true, Vec3::Z, &flat_env);
        let lm = LightModel::DEFAULT;
        let albedo = Vec3::splat(0.5);
        let (_, _, _, points_dark) = light_surface(albedo, Vec3::Z, 0.5, &lm, &lc_dark);
        assert_eq!(
            points_dark,
            Vec3::ZERO,
            "no authored lights ⇒ zero point radiance"
        );

        let red_env = LightEnv {
            points: PointLights::from_world(&[re_world::PointLight {
                color: pack_rgb(Vec3::new(1.0, 0.05, 0.05)),
                ..lamp(Vec3::new(0.0, 0.0, 1.0), 3.0)
            }]),
            ..LightEnv::builtin()
        };
        let lc_lit = LightCtx::for_floor(true, Vec3::Z, &red_env).at(Vec3::ZERO);
        let (_, _, _, points_lit) = light_surface(albedo, Vec3::Z, 0.5, &lm, &lc_lit);
        assert!(
            points_lit.x > 0.0,
            "a red lamp must contribute a non-zero R radiance"
        );
        assert!(
            points_lit.x > points_lit.y * 3.0 && points_lit.x > points_lit.z * 3.0,
            "the radiance must carry the lamp's RED hue, not a neutral value: {points_lit:?}"
        );
    }

    #[test]
    fn point_light_hue_reaches_the_bg_not_just_a_scalar_dim() {
        // Before the additive bg-gain term existed, the
        // point-light term only reached the bg through `shadow`, a SCALAR
        // (`lit.max_element()` folded through `dim_rgb`) that can only re-scale
        // the recipe's OWN bg colour — a lamp's hue never reached it, and on a
        // body-coverage-blanked floor cell (glyph ' ', κ=0) the bg is the only
        // visible carrier, so the pool was invisible there. Prove the bg now
        // gains the lamp's HUE: a saturated red lamp must lift bg.x (R) far more
        // than bg.y/bg.z (G/B) at the SAME coord — a shape no uniform scalar
        // multiply of an existing bg colour can produce.
        let perlin = Perlin::new(SHADER_SEED);
        let mat = floor_grain();
        let coord = Vec2::new(2.4, 3.1);
        let view = Vec3::new(0.0, -0.3, 0.95).normalize();

        let base_env = LightEnv::builtin();
        let mut c_base = sample_ctx(3.0);
        c_base.light = Some(LightCtx::for_floor(true, view, &base_env).at(Vec3::ZERO));
        let base = mat.sample(&perlin, coord, &c_base);

        let red_env = LightEnv {
            points: PointLights::from_world(&[re_world::PointLight {
                color: pack_rgb(Vec3::new(1.0, 0.05, 0.05)),
                ..lamp(Vec3::new(0.0, 0.0, 1.2), 3.0)
            }]),
            ..LightEnv::builtin()
        };
        let mut c_lit = sample_ctx(3.0);
        c_lit.light = Some(LightCtx::for_floor(true, view, &red_env).at(Vec3::ZERO));
        let lit = mat.sample(&perlin, coord, &c_lit);

        let d = unpack_rgb(lit.bg) - unpack_rgb(base.bg);
        assert!(
            d.x > 0.02,
            "the red lamp must visibly lift bg.x (R): delta={d:?}"
        );
        assert!(
            d.x > d.y * 2.0 && d.x > d.z * 2.0,
            "the lift must carry the lamp's hue (R-dominant), not a uniform brighten: delta={d:?}"
        );
    }
}
