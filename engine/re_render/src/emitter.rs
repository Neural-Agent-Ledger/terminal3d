//! `re_render::emitter` — particle emitters → the `Emitters` composite layer.
//!
//! The fast tier of the tiered-composite design: the particle pool is
//! integrated EVERY frame (cheap, must stay smooth even when near geometry is
//! static) and splatted into slot 3 of the [`CompositeStack`](re_core::CompositeStack) — the
//! `GlyphOnly` emitters layer that keeps the bg behind it.
//!
//! The whole module is pure compute: no terminal, no I/O, no per-frame heap
//! allocation. The particle pool is a fixed-capacity SoA with RING reuse
//! ([`ParticlePool`]); spawning consumes dead slots, so steady state allocates
//! nothing.
//!
//! ## Projection
//!
//! `splat_particles` projects a world particle into a `(col, row, depth)` triple
//! using the INVERSE of the raycaster's per-column projection (`crate::lib`):
//!
//! * camera basis `dir = (sin yaw, cos yaw)`, plane direction `(cos yaw, -sin yaw)`,
//! * `depth   = (p − cam) · dir`        — the fisheye-free PERPENDICULAR distance,
//!   exactly the value the raycaster stores in `Cell::depth`, so the occlusion test
//!   against `WorldOpaque.depth` is apples-to-apples,
//! * `lateral = (p − cam) · plane_dir`,
//! * `camera_x = (lateral / depth) / plane_scale ∈ [−1, 1]` ⇒
//!   `col = (camera_x + 1)/2 · cols`,
//! * `row = horizon − proj_plane_dist · (p.z − cam.z) / depth`.
//!
//! A particle behind the camera (`depth ≤ 0`), off-screen, or with a depth greater
//! than the opaque wall depth at its cell is skipped. Only live particles touch the
//! layer, so its `present` mask stays mostly false (the sparse claim).

use bevy::prelude::*;
use glam::{Vec3, Vec3Swizzles};
use re_core::{Cell, CellStyle, Layer, UnderlineStyle};
use re_world::{
    BoldMode, CameraPose, ColorJitter, ColorMode, Distribution, DustVisibility, EmitterBlend,
    EmitterKind, EmitterOrigin, EmitterSpec, GlyphPick, GlyphSource, MotionKind, PointLight,
    TrailSpec,
};

use crate::glyph_family::{family_by_name, GlyphFamily, GROWTH_DOT};
use crate::ripple;
use crate::{view_params, Fov};

/// Maximum lights an [`Emitter`] can bake for `visibility = lit` / `seek_light`.
/// Mirrors `ascii_shader::MAX_POINT_LIGHTS`'s spirit without depending on that
/// module — a handful of nearby lamps is plenty for a dust shaft or a firefly
/// swarm.
pub const MAX_EMITTER_LIGHTS: usize = 4;

/// One light baked into an [`Emitter`] at resolve time: world position, a
/// scalar brightness proxy (`luma`, from the light's linear colour × intensity),
/// and `1/radius²` (matching `ascii_shader::PointLightRt`'s attenuation shape,
/// reimplemented here so `re_render::emitter` does not need to depend on the
/// `ascii_shader` module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterLight {
    /// World-space position of the light.
    pub pos: Vec3,
    /// Scalar brightness proxy: linear colour luma × intensity.
    pub luma: f32,
    /// Precomputed `1/radius²` for the inverse-square falloff in [`light_at`].
    pub inv_radius2: f32,
}

/// Attenuated brightness of `lights` at `p` (no surface normal — dust is
/// volumetric, not shaded): `Σ luma / (1 + d²·inv_radius2)`. Pure.
#[inline]
pub fn light_at(lights: &[EmitterLight], p: Vec3) -> f32 {
    let mut acc = 0.0f32;
    for l in lights {
        let d2 = (l.pos - p).length_squared();
        acc += l.luma / (1.0 + d2 * l.inv_radius2);
    }
    acc
}

/// Perceptual luma (Rec. 601) of a packed `0x00RRGGBB` colour in `[0,1]`.
#[inline]
fn packed_luma(color: u32) -> f32 {
    let r = ((color >> 16) & 0xFF) as f32 / 255.0;
    let g = ((color >> 8) & 0xFF) as f32 / 255.0;
    let b = (color & 0xFF) as f32 / 255.0;
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Bake up to [`MAX_EMITTER_LIGHTS`] level-authored [`PointLight`]s into runtime
/// [`EmitterLight`]s, nearest-`origin`-first (so a cap always keeps the most
/// relevant lamps for THIS emitter, not just file order). Degenerate lights
/// (non-finite/non-positive radius or intensity) are dropped, matching
/// `ascii_shader::PointLights::from_world`'s contract.
pub fn nearest_lights(lights: &[PointLight], origin: Vec3) -> Vec<EmitterLight> {
    let mut candidates: Vec<&PointLight> = lights
        .iter()
        .filter(|l| {
            l.radius.is_finite()
                && l.radius > 0.0
                && l.intensity.is_finite()
                && l.intensity > 0.0
                && l.pos.is_finite()
        })
        .collect();
    candidates.sort_by(|a, b| {
        let da = (a.pos - origin).length_squared();
        let db = (b.pos - origin).length_squared();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates
        .into_iter()
        .take(MAX_EMITTER_LIGHTS)
        .map(|l| EmitterLight {
            pos: l.pos,
            luma: packed_luma(l.color) * l.intensity,
            inv_radius2: 1.0 / (l.radius * l.radius),
        })
        .collect()
}

// ───────────────────────────── colour ramp ──────────────────────────────────

/// A 2–3 stop colour ramp over a particle's NORMALIZED age `t ∈ [0,1]`. Stops are
/// `0xRRGGBB`; [`sample`](ColorRamp::sample) linearly interpolates the active segment.
/// Pure + `Copy` so an [`Emitter`] stays `Copy`.
/// Ramps carry at most this many stops (an authored `color_ramp` may hold 2-8).
pub const COLOR_RAMP_MAX_STOPS: usize = 8;

/// A fixed-capacity colour ramp over a particle's normalized age. See the
/// module-level doc comment above [`COLOR_RAMP_MAX_STOPS`] for the shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorRamp {
    /// Ordered `(age, 0xRRGGBB)` stops, `age` ascending in `[0,1]`.
    /// `count` of these are active; the rest repeat the last active stop so
    /// `sample`'s segment search never reads uninitialised-looking data.
    stops: [(f32, u32); COLOR_RAMP_MAX_STOPS],
    /// How many of `stops` are active (2..=8).
    count: u8,
}

impl ColorRamp {
    /// A two-stop ramp `birth → death` (the common ember/haze case).
    pub const fn two(birth: u32, death: u32) -> Self {
        ColorRamp {
            stops: [
                (0.0, birth),
                (1.0, death),
                (1.0, death),
                (1.0, death),
                (1.0, death),
                (1.0, death),
                (1.0, death),
                (1.0, death),
            ],
            count: 2,
        }
    }

    /// A three-stop ramp `birth → mid → death`.
    pub const fn three(birth: u32, mid: u32, death: u32) -> Self {
        ColorRamp {
            stops: [
                (0.0, birth),
                (0.5, mid),
                (1.0, death),
                (1.0, death),
                (1.0, death),
                (1.0, death),
                (1.0, death),
                (1.0, death),
            ],
            count: 3,
        }
    }

    /// An N-stop ramp (2..=8) evenly spaced over `[0,1]` (spec §4.5
    /// `color_ramp = 0xRRGGBB …`). Fewer than 2 stops falls back to a flat
    /// `two(c, c)`; more than [`COLOR_RAMP_MAX_STOPS`] are truncated (the spec
    /// parser/validator already enforces this range, so this is a safety net).
    pub fn from_stops(colors: &[u32]) -> Self {
        if colors.len() < 2 {
            let c = colors.first().copied().unwrap_or(0);
            return ColorRamp::two(c, c);
        }
        let n = colors.len().min(COLOR_RAMP_MAX_STOPS);
        let mut stops = [(1.0, colors[n - 1]); COLOR_RAMP_MAX_STOPS];
        for (i, stop) in stops.iter_mut().enumerate().take(n) {
            let age = i as f32 / (n - 1) as f32;
            *stop = (age, colors[i]);
        }
        ColorRamp {
            stops,
            count: n as u8,
        }
    }

    /// Return a copy with every stop's colour perturbed in HSL by `jitter`,
    /// using ONE consistent `(dh, ds, dl)` delta drawn from `seed` for every
    /// stop (spec §4.3 "Colour jitter, Brogue-style") — so the whole ramp
    /// shifts together (a coherently warmer/cooler variant), not stop-by-stop
    /// noise. Pure in `(self, jitter, seed)`.
    pub fn jittered(&self, jitter: ColorJitter, seed: u64) -> Self {
        let dh = signed_from(seed, 0xC001) * jitter.h_deg;
        let ds = signed_from(seed, 0xC002) * jitter.s;
        let dl = signed_from(seed, 0xC003) * jitter.l;
        let mut out = *self;
        for stop in out.stops.iter_mut().take(out.count as usize) {
            stop.1 = jitter_rgb_hsl(stop.1, dh, ds, dl);
        }
        out
    }

    /// The `0xRRGGBB` colour at normalized age `t`, clamped to `[0,1]`, by linear
    /// per-channel interpolation of the bracketing stops. Pure.
    pub fn sample(&self, t: f32) -> u32 {
        let t = t.clamp(0.0, 1.0);
        let n = self.count as usize;
        // Find the segment [i, i+1] containing t.
        let mut i = 0usize;
        while i + 1 < n && t > self.stops[i + 1].0 {
            i += 1;
        }
        let (t0, c0) = self.stops[i];
        let (t1, c1) = self.stops[(i + 1).min(n - 1)];
        let span = t1 - t0;
        let f = if span > f32::EPSILON {
            ((t - t0) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        lerp_rgb(c0, c1, f)
    }
}

/// Perturb a packed `0xRRGGBB` colour in HSL space: rotate hue by `dh` degrees,
/// scale saturation/lightness by `(1+ds)`/`(1+dl)` (clamped to `[0,1]`), then
/// convert back. A documented substitute for the design doc's Oklch colour
/// jitter — the runtime Oklab helpers do not exist yet, so
/// this stays in HSL, which is simple, self-contained, and gives the same
/// qualitative "coherent hue/tone drift per particle" result. See
/// `docs/EMITTERS.md` "Colour jitter is HSL, not Oklch (yet)".
#[inline]
fn jitter_rgb_hsl(rgb: u32, dh: f32, ds: f32, dl: f32) -> u32 {
    let (h, s, l) = rgb_to_hsl(rgb);
    let h2 = (h + dh).rem_euclid(360.0);
    let s2 = (s * (1.0 + ds)).clamp(0.0, 1.0);
    let l2 = (l * (1.0 + dl)).clamp(0.0, 1.0);
    hsl_to_rgb(h2, s2, l2)
}

/// `0xRRGGBB` → `(hue degrees, saturation 0..1, lightness 0..1)`.
fn rgb_to_hsl(rgb: u32) -> (f32, f32, f32) {
    let r = ((rgb >> 16) & 0xFF) as f32 / 255.0;
    let g = ((rgb >> 8) & 0xFF) as f32 / 255.0;
    let b = (rgb & 0xFF) as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let delta = max - min;
    if delta < 1e-6 {
        return (0.0, 0.0, l);
    }
    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };
    // Which channel is the max: compare the channels directly (not against the
    // already-rounded `max`) so this never becomes a strict float-equality check.
    let h = if r >= g && r >= b {
        60.0 * (((g - b) / delta).rem_euclid(6.0))
    } else if g >= r && g >= b {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    (h, s, l)
}

/// `(hue degrees, saturation 0..1, lightness 0..1)` → `0xRRGGBB`.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> u32 {
    if s <= 1e-6 {
        let v = (l * 255.0).round().clamp(0.0, 255.0) as u32;
        return (v << 16) | (v << 8) | v;
    }
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h.rem_euclid(360.0) / 60.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c * 0.5;
    let to_byte = |v: f32| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u32;
    (to_byte(r1) << 16) | (to_byte(g1) << 8) | to_byte(b1)
}

/// Linear interpolation between two `0xRRGGBB` colours at `f ∈ [0,1]`.
#[inline]
fn lerp_rgb(a: u32, b: u32, f: f32) -> u32 {
    let f = f.clamp(0.0, 1.0);
    let lerp = |x: u32, y: u32| {
        let xf = x as f32;
        let yf = y as f32;
        (xf + (yf - xf) * f).round().clamp(0.0, 255.0) as u32
    };
    let r = lerp((a >> 16) & 0xFF, (b >> 16) & 0xFF);
    let g = lerp((a >> 8) & 0xFF, (b >> 8) & 0xFF);
    let bl = lerp(a & 0xFF, b & 0xFF);
    (r << 16) | (g << 8) | bl
}

// ─────────────────────────────── emitter ────────────────────────────────────

/// Where an [`Emitter`] spawns particles from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    /// A world-space point (cells). Particles live in world space and project via
    /// the camera (depth-tested, occluded by geometry).
    World(Vec3),
    /// A fixed SCREEN cell `(col, row)`. The particle is pinned to the viewport (an
    /// overlay spark); it bypasses world projection and is never occluded.
    Screen(u16, u16),
}

/// A particle SOURCE: a spawn rate, a lifetime, an initial velocity field, an
/// acceleration (gravity / buoyancy), a [`GlyphFamily`] (members chosen by age),
/// a [`ColorRamp`] (fg over age) and a blend. Pure data.
///
/// Not `Copy`: [`nearby_lights`](Self::nearby_lights) is a `Vec`. Every
/// caller builds a `Vec<Emitter>` via `.map(Emitter::from_spec).collect()`,
/// which never relies on `Copy`.
#[derive(Clone, Debug)]
pub struct Emitter {
    /// Where particles are born.
    pub origin: Anchor,
    /// Particles spawned per second (deterministic, fractional accumulated).
    pub rate: f32,
    /// Optional one-shot BURST count: `Some(n)` emits exactly `n` particles when the
    /// emitter first ticks, then never again (and `rate` is ignored). `None` = a
    /// continuous stream at `rate`.
    pub burst: Option<u32>,
    /// Seconds a particle lives before it is culled (mean; see `lifetime_spread`).
    pub lifetime: f32,
    /// Symmetric lifetime spread (`±`, uniform) — variety in how long motes persist.
    pub lifetime_spread: f32,
    /// Mean initial velocity (cells/sec). For a spec-built emitter this is
    /// `direction.normalize() * speed`; the cone spread is applied per spawn.
    pub vel: Vec3,
    /// Symmetric per-axis spread added to `vel` (`±vel_spread`, uniform). Legacy
    /// demo emitters use this; spec-built emitters drive spread via the CONE
    /// (`direction` + `spread_deg`) and `speed_spread` instead.
    pub vel_spread: Vec3,
    /// Emission CONE: the mean direction (unit) the spec authored. Spawn velocity is
    /// this rotated within `spread_rad`, scaled by `speed ± speed_spread`. Zero ⇒ the
    /// emitter uses the legacy `vel`/`vel_spread` field instead (demo path).
    pub direction: Vec3,
    /// Cone HALF-ANGLE (radians) spawn directions scatter within. `0` = a pencil beam.
    pub spread_rad: f32,
    /// Mean speed along the cone direction (cells/sec).
    pub speed: f32,
    /// Symmetric speed spread (`±`, uniform).
    pub speed_spread: f32,
    /// Constant acceleration (cells/sec²) — gravity (−z) or buoyancy (+z).
    pub accel: Vec3,
    /// Base size at birth (drives the Growth-family glyph index at `t=0`).
    pub size: f32,
    /// Size scale over age: effective size at `t=1` is `size * growth`. Drives the
    /// Growth glyph index up (`>1`) or down (`<1`) the family as the particle ages.
    pub growth: f32,
    /// Path JITTER amplitude (cells/sec of deterministic turbulence). `0` = smooth.
    pub path_jitter: f32,
    /// Curl/SWIRL strength (cells/sec) about the vertical axis (counter-clockwise for
    /// positive, viewed from above). `0` = no swirl.
    pub swirl: f32,
    /// Deterministic base SEED. Each particle's path is reproducible from
    /// `(seed, birth_index)` — see [`tick_emitters`]'s per-particle hash.
    pub seed: u64,
    /// When (if ever) to set the terminal BOLD attribute on the splat.
    pub bold: BoldMode,
    /// Glyph family; a particle's glyph is picked by its size/age (Growth
    /// small→large reads as a particle blooming then fading).
    pub family: GlyphFamily,
    /// Foreground colour over normalized age.
    pub ramp: ColorRamp,
    /// The three authored colour stops + how to apply them (drives `RandomPick`,
    /// which the per-particle seed selects at birth; `RampOverAge`/`Fixed` use
    /// [`ramp`](Self::ramp) directly).
    pub colors: [u32; 3],
    /// How `colors` combines with `ramp` to pick a particle's colour.
    pub color_mode: ColorMode,
    /// Hard per-emitter live-particle cap (spawning is skipped at the ceiling).
    pub max_particles: u32,
    /// Composite blend for the splat (`GlyphOnly` keeps the bg behind it).
    pub blend: re_core::LayerBlend,
    /// The emitter's high-level rendering KIND. [`EmitterKind::Standard`] splats a
    /// glyph into the `emitters` layer; [`EmitterKind::GodRays`] splats a COLOUR-ONLY,
    /// alpha-carrying haze cell (no glyph) into the `godrays` layer instead — so the
    /// two coexist in one scene without the shafts erasing glyphs. Carried per
    /// particle in the pool so the splat routes each to the right layer.
    pub kind: EmitterKind,

    // ── VARIETY & RANDOMISATION — mirrors `EmitterSpec`'s fields one for
    // one; see that type's docs for the semantics. Every default below
    // reproduces the baseline (no variety) behaviour exactly.
    /// Bulk motion model (`MotionKind::Ballistic` = today's vel/accel path).
    pub motion: MotionKind,
    /// Lifetime distribution override (`None` = plain `lifetime ± lifetime_spread`).
    pub lifetime_dist: Option<Distribution>,
    /// Speed distribution override (`None` = plain `speed ± speed_spread`).
    pub speed_dist: Option<Distribution>,
    /// Birth-size distribution override (`None` = plain `size`).
    pub size_dist: Option<Distribution>,
    /// Scales every spread quantity (`1.0` = unchanged).
    pub randomize: f32,
    /// Per-particle HSL colour jitter, re-rolled once at birth.
    pub color_jitter: Option<ColorJitter>,
    /// Whether the glyph is age-driven (`Sequential`) or a fixed per-particle
    /// pick (`Random`).
    pub glyph_pick: GlyphPick,
    /// Where the glyph sequence comes from (`Family` = today's family/`glyph_set`).
    pub glyph_source: GlyphSource,
    /// A fading motion trail behind the particle (`None` = no trail).
    pub trail: Option<TrailSpec>,
    /// Floor-bounce restitution (`None` = no bounce). The floor height is
    /// [`floor_z`](Self::floor_z).
    pub bounce: Option<f32>,
    /// The world height a bouncing particle reflects off (baked from the
    /// emitter's `World` origin at resolve time; meaningless without `bounce`).
    pub floor_z: f32,
    /// Whether the particle always renders or only where lit.
    pub visibility: DustVisibility,
    /// Lights baked in at resolve time for `visibility = Lit` / `Flock.seek_light`.
    /// Empty unless the emitter was built via [`Emitter::from_spec_lit`].
    pub nearby_lights: Vec<EmitterLight>,
}

/// Resolve an [`EmitterSpec`]'s `family` NAME (e.g. `"growth_dot"`) into a live
/// [`GlyphFamily`], falling back to [`GROWTH_DOT`] for an unknown name. The String→
/// family bridge the spec contract defines (`re_world` carries only the name).
pub fn family_by_name_or_default(name: &str) -> GlyphFamily {
    family_by_name(name).unwrap_or(GROWTH_DOT)
}

/// Build a [`GlyphFamily`] from an explicit authored glyph sequence (the spec's
/// `glyph_set` escape hatch). The slice must outlive the family; an emitter resolves
/// this ONCE at level load (not per frame), so leaking a small `Vec<char>` to a
/// `'static` slice here is a bounded, one-time cost — never in the hot path. Empty
/// or all-unreliable input falls back to [`GROWTH_DOT`].
fn family_from_glyph_set(glyphs: &[char]) -> GlyphFamily {
    if glyphs.is_empty() {
        return GROWTH_DOT;
    }
    // One-time leak: an authored glyph_set is fixed for the level's lifetime, so a
    // `'static` slice is correct and the cost is paid once per emitter at load.
    let leaked: &'static [char] = Box::leak(glyphs.to_vec().into_boxed_slice());
    GlyphFamily {
        name: "glyph_set",
        kind: crate::glyph_family::FamilyKind::Growth,
        glyphs: leaked,
    }
}

impl Emitter {
    /// A rising EMBER COLUMN at `origin`: warm, buoyant, GROWTH_DOT glyphs fading
    /// from bright orange to dim red. A good visible demo source.
    pub fn ember_column(origin: Vec3) -> Self {
        Emitter {
            origin: Anchor::World(origin),
            // Playtest: motion read too fast & too dense → trim the rate (24→16) and
            // halve the buoyant rise (0.9→0.45) for a calmer, floatier ember stream.
            rate: 16.0,
            burst: None,
            // A touch longer mean life + a real ± spread so motes don't all expire in
            // lockstep (stronger per-particle variation, complaint B).
            lifetime: 2.6,
            lifetime_spread: 0.9,
            // Halved vertical drift; the lateral/vertical spreads are widened so each
            // ember diverges from its neighbour instead of marching up identically.
            vel: Vec3::new(0.0, 0.0, 0.45),
            vel_spread: Vec3::new(0.55, 0.55, 0.4),
            direction: Vec3::ZERO, // legacy demo: use vel/vel_spread, not the cone.
            spread_rad: 0.0,
            speed: 0.0,
            speed_spread: 0.0,
            accel: Vec3::new(0.0, 0.0, 0.18), // gentler buoyancy (was 0.35)
            size: 1.0,
            growth: 1.0,
            // Turbulence + a slight swirl so neighbouring embers trace visibly
            // different, wandering paths (complaint B) — still deterministic per seed.
            path_jitter: 0.4,
            swirl: 0.5,
            seed: 0x0E3B_7C0D,
            bold: BoldMode::Off,
            family: GROWTH_DOT,
            ramp: ColorRamp::three(0xFFD27F, 0xFF7A1A, 0x6E1200),
            colors: [0xFFD27F, 0xFF7A1A, 0x6E1200],
            color_mode: ColorMode::RampOverAge,
            max_particles: DEFAULT_POOL_CAP as u32,
            blend: re_core::LayerBlend::GlyphOnly,
            kind: EmitterKind::Standard,
            motion: MotionKind::Ballistic,
            lifetime_dist: None,
            speed_dist: None,
            size_dist: None,
            randomize: 1.0,
            color_jitter: None,
            glyph_pick: GlyphPick::Sequential,
            glyph_source: GlyphSource::Family,
            trail: None,
            bounce: None,
            floor_z: origin.z,
            visibility: DustVisibility::Always,
            nearby_lights: Vec::new(),
        }
    }

    /// A slow DRIFTING HAZE around `origin`: faint cool motes drifting sideways,
    /// nearly weightless. The second demo source.
    pub fn drifting_haze(origin: Vec3) -> Self {
        Emitter {
            origin: Anchor::World(origin),
            // Playtest: calmer & sparser → trim the rate (14→9). Haze should barely
            // move, so the drift velocity is roughly halved.
            rate: 9.0,
            burst: None,
            // Longer-lived motes with a wide ± life spread so the cloud breathes
            // instead of pulsing uniformly (complaint B).
            lifetime: 5.0,
            lifetime_spread: 1.6,
            // Halved sideways drift; widened per-axis spread so each mote wanders its
            // own way rather than sliding in a tight identical sheet.
            vel: Vec3::new(0.09, 0.0, 0.025),
            vel_spread: Vec3::new(0.55, 0.55, 0.22),
            direction: Vec3::ZERO,
            spread_rad: 0.0,
            speed: 0.0,
            speed_spread: 0.0,
            accel: Vec3::new(0.0, 0.0, -0.01),
            size: 1.0,
            growth: 1.0,
            // Gentle turbulence + a slow swirl give the haze an organic, diverging
            // drift (complaint B) — deterministic per seed.
            path_jitter: 0.5,
            swirl: 0.35,
            seed: 0x4A2E_51A2_u64,
            bold: BoldMode::Off,
            family: crate::glyph_family::HAZE_DOT,
            ramp: ColorRamp::two(0x9FB8D8, 0x2A3A52),
            colors: [0x9FB8D8, 0x4A7AA8, 0x2A3A52],
            color_mode: ColorMode::RampOverAge,
            max_particles: DEFAULT_POOL_CAP as u32,
            blend: re_core::LayerBlend::GlyphOnly,
            kind: EmitterKind::Standard,
            motion: MotionKind::Ballistic,
            lifetime_dist: None,
            speed_dist: None,
            size_dist: None,
            randomize: 1.0,
            color_jitter: None,
            glyph_pick: GlyphPick::Sequential,
            glyph_source: GlyphSource::Family,
            trail: None,
            bounce: None,
            floor_z: origin.z,
            visibility: DustVisibility::Always,
            nearby_lights: Vec::new(),
        }
    }

    /// A warm GOD-RAYS / light-shaft emitter at `origin`: a colour-only volumetric
    /// haze (no glyphs) cast along `dir`. Its particles TINT whatever they cover
    /// toward a warm yellow→white via the `godrays` layer's `ColorMix` blend, fading
    /// with age (the alpha the splat packs into the high byte falls off near death),
    /// so the shaft reads brightest near the source and dissolves down-beam. A good
    /// visible demo source for the GodRays kind.
    pub fn god_rays(origin: Vec3, dir: Vec3) -> Self {
        Emitter {
            origin: Anchor::World(origin),
            // A steady, fairly dense stream so the shaft reads as a continuous body of
            // light rather than discrete motes (the glyph never shows — only colour).
            rate: 36.0,
            burst: None,
            lifetime: 3.0,
            lifetime_spread: 0.6,
            vel: Vec3::ZERO,
            vel_spread: Vec3::ZERO,
            direction: dir.normalize_or_zero(),
            // A narrow shaft: a few degrees of spread so the beam has soft edges.
            spread_rad: 8.0_f32.to_radians(),
            speed: 1.1,
            speed_spread: 0.25,
            accel: Vec3::ZERO,
            size: 1.0,
            growth: 1.0,
            path_jitter: 0.25,
            swirl: 0.0,
            seed: 0x5A17_BEAD,
            bold: BoldMode::Off,
            // The family is irrelevant (a god-rays splat draws no glyph) but must be
            // a valid family for the pool — GROWTH_DOT is the neutral default.
            family: GROWTH_DOT,
            // Warm light ramp: bright near-white core → warm gold → deep amber. The
            // splat reads this as the TINT colour, then fades the ALPHA with age.
            ramp: ColorRamp::three(0xFFF6E0, 0xFFD27A, 0xC8841E),
            colors: [0xFFF6E0, 0xFFD27A, 0xC8841E],
            color_mode: ColorMode::RampOverAge,
            max_particles: DEFAULT_POOL_CAP as u32,
            // The blend on the Emitter is unused for GodRays (it splats into the
            // dedicated `godrays` ColorMix layer); set ColorMix for documentation.
            blend: re_core::LayerBlend::ColorMix,
            kind: EmitterKind::GodRays,
            motion: MotionKind::Ballistic,
            lifetime_dist: None,
            speed_dist: None,
            size_dist: None,
            randomize: 1.0,
            color_jitter: None,
            glyph_pick: GlyphPick::Sequential,
            glyph_source: GlyphSource::Family,
            trail: None,
            bounce: None,
            floor_z: origin.z,
            visibility: DustVisibility::Always,
            nearby_lights: Vec::new(),
        }
    }

    /// Resolve an authored [`EmitterSpec`] into a live runtime [`Emitter`].
    ///
    /// The String→render bridge the spec contract defines: validates the spec, then
    /// maps every authored control into runtime behaviour —
    /// * `origin` → [`Anchor`] (World/Screen),
    /// * `family` NAME (or `glyph_set` escape hatch) → [`GlyphFamily`],
    /// * `colors` + `color_mode` → a [`ColorRamp`] (RampOverAge = 3-stop, Fixed =
    ///   1-stop, RandomPick = a per-particle pick the seed drives),
    /// * `direction` + `spread_deg` + `speed` + `speed_spread` → the spawn CONE,
    /// * `size`/`growth` → the Growth-glyph index over age,
    /// * `bold`, `path_jitter`, `swirl`, `seed`, `lifetime(_spread)`, `accel`,
    ///   `max_particles`, `burst`, `rate`, `blend` → their runtime fields.
    ///
    /// Deterministic: same spec → same `Emitter` → identical particle field every run.
    pub fn from_spec(spec: &EmitterSpec) -> Self {
        Self::from_spec_lit(spec, &[])
    }

    /// [`from_spec`](Self::from_spec), plus baking the [`MAX_EMITTER_LIGHTS`]
    /// nearest `lights` in (`visibility = lit` and `Flock.seek_light` need SOME
    /// light source to react to). `from_spec` forwards here with an empty
    /// light slice — the baseline behaviour for every existing caller — so
    /// wiring the level's real lights through this path is an additive,
    /// opt-in step for whatever calls into `re_game::state_bridge`/`flow.rs`.
    pub fn from_spec_lit(spec: &EmitterSpec, lights: &[PointLight]) -> Self {
        let spec = spec.clone().validated();

        let origin = match spec.origin {
            EmitterOrigin::World(p) => Anchor::World(p),
            EmitterOrigin::Screen { col, row } => {
                Anchor::Screen(col.max(0.0) as u16, row.max(0.0) as u16)
            }
        };

        let family = match &spec.glyph_set {
            Some(g) => family_from_glyph_set(g),
            None => family_by_name_or_default(&spec.family),
        };

        let ramp = if let Some(stops) = &spec.color_ramp {
            // `color_ramp` (spec §4.5) overrides `colors`/`color_mode` entirely.
            ColorRamp::from_stops(stops)
        } else {
            match spec.color_mode {
                ColorMode::RampOverAge => {
                    ColorRamp::three(spec.colors[0], spec.colors[1], spec.colors[2])
                }
                // Fixed: a single tint for the whole life. A two-stop ramp birth==death.
                ColorMode::Fixed => ColorRamp::two(spec.colors[0], spec.colors[0]),
                // RandomPick: the per-particle seed picks one stop at birth and holds
                // it; the base ramp is a (birth==death) of stop 0 as a sane fallback
                // for any path that ignores the per-particle pick.
                ColorMode::RandomPick => ColorRamp::two(spec.colors[0], spec.colors[0]),
            }
        };

        let world_origin = match spec.origin {
            EmitterOrigin::World(p) => p,
            EmitterOrigin::Screen { .. } => Vec3::ZERO,
        };
        let floor_z = world_origin.z;
        let nearby_lights = nearest_lights(lights, world_origin);

        // The cone direction (unit). A zero/degenerate authored direction ⇒ straight
        // up, treated as omni by the wide spread the validator already clamped.
        let dir = spec.direction.normalize_or_zero();
        let direction = if dir == Vec3::ZERO { Vec3::Z } else { dir };

        // Bold cue: `On` lights every splat; the proximity/age cues are evaluated per
        // splat. Carried as the `BoldMode` so the splat decides.
        let blend = match spec.blend {
            EmitterBlend::KeepBg => re_core::LayerBlend::GlyphOnly,
            EmitterBlend::Opaque => re_core::LayerBlend::Opaque,
        };

        Emitter {
            origin,
            rate: spec.rate,
            burst: spec.burst,
            lifetime: spec.lifetime,
            lifetime_spread: spec.lifetime_spread,
            // The cone fully drives spawn velocity for a spec emitter; the legacy
            // per-axis `vel`/`vel_spread` are zeroed so they don't double-apply.
            vel: Vec3::ZERO,
            vel_spread: Vec3::ZERO,
            direction,
            spread_rad: spec.spread_deg.to_radians(),
            speed: spec.speed,
            speed_spread: spec.speed_spread,
            accel: spec.accel,
            size: spec.size,
            growth: spec.growth,
            path_jitter: spec.path_jitter,
            swirl: spec.swirl,
            seed: spec.seed,
            bold: spec.bold,
            family,
            ramp,
            colors: spec.colors,
            color_mode: spec.color_mode,
            max_particles: spec.max_particles,
            blend,
            kind: spec.kind,
            motion: spec.motion,
            lifetime_dist: spec.lifetime_dist,
            speed_dist: spec.speed_dist,
            size_dist: spec.size_dist,
            randomize: spec.randomize,
            color_jitter: spec.color_jitter,
            glyph_pick: spec.glyph_pick,
            glyph_source: spec.glyph_source.clone(),
            trail: spec.trail,
            bounce: spec.bounce,
            floor_z,
            visibility: spec.visibility,
            nearby_lights,
        }
    }
}

// ──────────────────────────── particle pool ─────────────────────────────────

/// Default fixed pool capacity. Sized so the two demo emitters at their rates
/// never exhaust it (24+14 particles/s × ~4 s lifetime ≈ 150 live; 4096 leaves
/// generous headroom for a level with several emitters).
pub const DEFAULT_POOL_CAP: usize = 4096;

/// A fixed-capacity Structure-of-Arrays particle pool with RING reuse. Spawning
/// scans for a dead slot starting at a rolling cursor; a culled particle frees its
/// slot for the next spawn. After warm-up the `Vec`s never grow — ZERO per-frame
/// allocation (spec §7).
#[derive(Resource, Clone, Debug)]
pub struct ParticlePool {
    /// Per-particle BALLISTIC position (the deterministic turbulence/swirl offset is
    /// added on top at splat time — see [`particle_world_pos`] — so the stored base
    /// path stays clean and the noise never compounds frame-to-frame).
    pos: Vec<Vec3>,
    vel: Vec<Vec3>,
    /// Per-particle constant acceleration (copied from its emitter at spawn) so a
    /// scene with mixed fields (buoyant embers + sinking haze) integrates each
    /// particle correctly.
    accel: Vec<Vec3>,
    age: Vec<f32>,
    life: Vec<f32>,
    /// Per-particle ramp colour source (one per live particle — copied at spawn so
    /// the splat needs no emitter back-reference).
    ramp: Vec<ColorRamp>,
    /// Per-particle glyph family (copied at spawn).
    family: Vec<GlyphFamily>,
    /// Per-particle DETERMINISTIC seed = `hash(emitter.seed, birth_index)`. Drives
    /// the path turbulence/swirl phase + the `RandomPick` colour — so a particle's
    /// whole trajectory + look is reproducible from `(emitter.seed, birth_index)` and
    /// never re-rolls frame to frame (the user's stability requirement).
    seed: Vec<u64>,
    /// Per-particle base size at birth + size growth scale (drive the Growth glyph).
    size0: Vec<f32>,
    growth: Vec<f32>,
    /// Per-particle path-turbulence amplitude + swirl strength (copied at spawn).
    jitter: Vec<f32>,
    swirl: Vec<f32>,
    /// Per-particle bold cue (copied at spawn; the splat resolves it).
    bold: Vec<BoldMode>,
    /// Per-particle FIXED foreground from `RandomPick` (a stop chosen at birth), or
    /// `NO_FG_PICK` to mean "use the ramp". Holds for the particle's life.
    fg_pick: Vec<u32>,
    /// Per-particle GOD-RAYS flag (copied at spawn from the emitter kind): `true` ⇒
    /// the splat routes this particle into the `godrays` layer as a colour-only,
    /// alpha-carrying haze cell (no glyph); `false` ⇒ a standard glyph splat into the
    /// `emitters` layer. Carried per particle so a mixed scene splats correctly.
    godray: Vec<bool>,
    /// Per-particle RIPPLE flag (copied at spawn from `em.motion` — `true` iff the
    /// owning emitter's motion is [`MotionKind::Ripple`]). Drives the `curly`
    /// colored-underline hem at splat time, matching `godray`'s
    /// "copied at spawn, no emitter back-reference needed at splat" pattern.
    ripple_hem: Vec<bool>,
    /// Per-particle owning emitter index (so a cull can decrement that emitter's live
    /// count for the `max_particles` cap). `u16::MAX` = unowned / legacy hand-placed.
    owner: Vec<u16>,
    alive: Vec<bool>,
    /// Rolling allocation cursor for the ring scan.
    cursor: usize,
    /// Fractional spawn accumulator per emitter index (carries sub-particle
    /// fractions of `rate·dt` across frames so the average spawn rate is exact).
    spawn_acc: Vec<f32>,
    /// Monotonic birth counter per emitter index — the `birth_index` half of each
    /// particle's deterministic seed (so successive particles trace distinct paths).
    birth: Vec<u64>,
    /// Whether each BURST emitter has already fired its one-shot (so it never repeats).
    burst_fired: Vec<bool>,
    /// Per-emitter live-particle count (enforces `Emitter::max_particles` without a
    /// full scan each spawn).
    live_per_emitter: Vec<u32>,

    // ── VARIETY & RANDOMISATION ──────────────────────────────────────────
    /// Raw Lorenz-attractor state (only meaningful for `MotionKind::Lorenz`
    /// particles — see [`tick_emitters`]). Seeded at spawn, integrated in place.
    lorenz: Vec<Vec3>,
    /// Fixed family glyph INDEX chosen at birth (`GlyphPick::Random`), or
    /// [`NO_FIXED_GLYPH`] to mean "use the age-driven pick" (the baseline
    /// behaviour).
    glyph_fixed: Vec<i16>,
    /// A literal glyph codepoint that BYPASSES the family entirely (`0` = none),
    /// used by `GlyphSource::Text`.
    literal_glyph: Vec<u32>,
    /// Birth-time visibility factor in `[0,1]` (`1.0` = always visible; a
    /// `DustVisibility::Lit` particle bakes a light sample here once at spawn —
    /// see [`EmitterSpec::visibility`] docs for why birth-time, not per-frame).
    vis: Vec<f32>,
    /// Trail ring buffer (last [`TRAIL_CAP`] positions, most-recent first) and
    /// how many of them are filled so far. Unused (n stays 0) unless the owning
    /// emitter authored a `trail`.
    trail_buf: Vec<[Vec3; TRAIL_CAP]>,
    trail_n: Vec<u8>,
}

/// Sentinel [`ParticlePool::glyph_fixed`] value meaning "no fixed pick".
const NO_FIXED_GLYPH: i16 = -1;

/// Trail ring capacity: "ring of the last K = 3-6 positions";
/// `TrailSpec::count` is clamped to this range.
pub const TRAIL_CAP: usize = 6;

/// Sentinel `fg_pick` meaning "no fixed pick — sample the ramp by age".
const NO_FG_PICK: u32 = 0xFFFF_FFFF;

impl ParticlePool {
    /// A pool of `cap` dead slots. `cap` is the hard ceiling — the pool never grows.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        ParticlePool {
            pos: vec![Vec3::ZERO; cap],
            vel: vec![Vec3::ZERO; cap],
            accel: vec![Vec3::ZERO; cap],
            age: vec![0.0; cap],
            life: vec![0.0; cap],
            ramp: vec![ColorRamp::two(0, 0); cap],
            family: vec![GROWTH_DOT; cap],
            seed: vec![0; cap],
            size0: vec![1.0; cap],
            growth: vec![1.0; cap],
            jitter: vec![0.0; cap],
            swirl: vec![0.0; cap],
            bold: vec![BoldMode::Off; cap],
            fg_pick: vec![NO_FG_PICK; cap],
            godray: vec![false; cap],
            ripple_hem: vec![false; cap],
            owner: vec![u16::MAX; cap],
            alive: vec![false; cap],
            cursor: 0,
            spawn_acc: Vec::new(),
            birth: Vec::new(),
            burst_fired: Vec::new(),
            live_per_emitter: Vec::new(),
            lorenz: vec![Vec3::ZERO; cap],
            glyph_fixed: vec![NO_FIXED_GLYPH; cap],
            literal_glyph: vec![0; cap],
            vis: vec![1.0; cap],
            trail_buf: vec![[Vec3::ZERO; TRAIL_CAP]; cap],
            trail_n: vec![0; cap],
        }
    }

    /// Fixed pool capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.alive.len()
    }

    /// Number of currently-live particles (a scan — used by tests / debug).
    pub fn live_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Find the next dead slot starting at `cursor`, wrapping once. `None` if the
    /// pool is full (the spawn is then dropped — a hard cap, never a realloc).
    #[inline]
    fn next_free(&mut self) -> Option<usize> {
        let cap = self.capacity();
        for k in 0..cap {
            let i = (self.cursor + k) % cap;
            if !self.alive[i] {
                self.cursor = (i + 1) % cap;
                return Some(i);
            }
        }
        None
    }

    /// Activate a slot with a fresh particle. Internal — called by [`tick_emitters`].
    /// Returns `true` if a slot was free (the spawn took); `false` if the pool was
    /// full (a hard cap, never a realloc).
    #[inline]
    fn spawn(&mut self, p: SpawnParticle) -> bool {
        if let Some(i) = self.next_free() {
            self.pos[i] = p.pos;
            self.vel[i] = p.vel;
            self.accel[i] = p.accel;
            self.age[i] = 0.0;
            self.life[i] = p.life.max(1e-3);
            self.ramp[i] = p.ramp;
            self.family[i] = p.family;
            self.seed[i] = p.seed;
            self.size0[i] = p.size0;
            self.growth[i] = p.growth;
            self.jitter[i] = p.jitter;
            self.swirl[i] = p.swirl;
            self.bold[i] = p.bold;
            self.fg_pick[i] = p.fg_pick;
            self.godray[i] = p.godray;
            self.ripple_hem[i] = p.ripple_hem;
            self.owner[i] = p.owner;
            self.lorenz[i] = p.lorenz0;
            self.glyph_fixed[i] = p.glyph_fixed;
            self.literal_glyph[i] = p.literal_glyph;
            self.vis[i] = p.vis;
            self.trail_buf[i] = [p.pos; TRAIL_CAP];
            self.trail_n[i] = 0;
            self.alive[i] = true;
            true
        } else {
            false
        }
    }

    /// Ensure the per-emitter bookkeeping vectors have one slot per emitter.
    #[inline]
    fn ensure_acc(&mut self, emitter_count: usize) {
        if self.spawn_acc.len() < emitter_count {
            self.spawn_acc.resize(emitter_count, 0.0);
        }
        if self.birth.len() < emitter_count {
            self.birth.resize(emitter_count, 0);
        }
        if self.burst_fired.len() < emitter_count {
            self.burst_fired.resize(emitter_count, false);
        }
        if self.live_per_emitter.len() < emitter_count {
            self.live_per_emitter.resize(emitter_count, 0);
        }
    }
}

/// The per-particle spawn payload (a small `Copy` struct so [`ParticlePool::spawn`]
/// stays a single argument and the call sites read field-by-field).
#[derive(Clone, Copy)]
struct SpawnParticle {
    pos: Vec3,
    vel: Vec3,
    accel: Vec3,
    life: f32,
    ramp: ColorRamp,
    family: GlyphFamily,
    seed: u64,
    size0: f32,
    growth: f32,
    jitter: f32,
    swirl: f32,
    bold: BoldMode,
    fg_pick: u32,
    godray: bool,
    ripple_hem: bool,
    owner: u16,
    lorenz0: Vec3,
    glyph_fixed: i16,
    literal_glyph: u32,
    vis: f32,
}

impl Default for ParticlePool {
    fn default() -> Self {
        ParticlePool::with_capacity(DEFAULT_POOL_CAP)
    }
}

// ─────────────────────────── deterministic RNG ──────────────────────────────

/// A tiny SplitMix64 — deterministic, no heap, matching the wrapping-mul hash style
/// the shader uses. Seeded per tick from a fixed base + the frame index so the
/// particle field is reproducible (golden-frame safe) yet varies frame to frame.
#[derive(Clone, Copy, Debug)]
pub struct ParticleRng {
    state: u64,
}

impl ParticleRng {
    /// Seed the stream. Same seed → identical particle positions every run.
    #[inline]
    pub fn seed(seed: u64) -> Self {
        ParticleRng {
            // Avoid the all-zero fixed point.
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A uniform `f32` in `[-1, 1)`.
    #[inline]
    pub fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

// ──────────────────── deterministic per-particle path field ──────────────────
//
// PATH RANDOMISATION (user requirement): a particle's jitter/swirl path must be
// reproducible from `(emitter.seed, birth_index)` and MUST NOT re-roll each frame.
// We satisfy this by giving every particle a fixed per-particle seed
// (`hash64(emitter.seed, birth_index)`) at spawn, then computing the turbulence as a
// PURE FUNCTION of `(particle_seed, age)` — no fresh RNG per tick. The integration
// only advances `age`; the randomness comes entirely from the fixed seed, so the
// same particle traces the same path every run and does not noise frame-to-frame.

/// Mix two `u64`s into one deterministic seed (SplitMix64 finaliser on a combine).
#[inline]
fn hash64(a: u64, b: u64) -> u64 {
    let mut z = a
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(b.wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic `f32 ∈ [-1, 1]` from a particle seed + an integer phase index.
#[inline]
fn signed_from(seed: u64, phase: u64) -> f32 {
    let h = hash64(seed, phase);
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// The DETERMINISTIC turbulence displacement (world cells) for a particle of fixed
/// `seed` at continuous `age`, scaled by `jitter` (amplitude). A function of
/// `(seed, age)` ONLY — same inputs, same vector, every run + every frame. Built by
/// smoothly interpolating per-second value-noise samples keyed by the seed, so the
/// drift is continuous (no jumps) yet fully reproducible. Returns `ZERO` for
/// `jitter == 0` (the smooth-path fast exit).
#[inline]
fn turbulence(seed: u64, age: f32, jitter: f32) -> Vec3 {
    if jitter <= 0.0 {
        return Vec3::ZERO;
    }
    // Value-noise: sample a pseudo-random vector at each integer second and smoothly
    // (smoothstep) interpolate. The seed offsets the per-axis streams so x/y/z drift
    // independently. A slow rate (≈0.7 Hz) reads as a gentle wander, not a buzz.
    let t = age * 0.7;
    let i0 = t.floor() as i64 as u64;
    let i1 = i0.wrapping_add(1);
    let f = t - t.floor();
    let s = f * f * (3.0 - 2.0 * f); // smoothstep
    let axis = |off: u64| -> f32 {
        let a = signed_from(seed ^ off, i0);
        let b = signed_from(seed ^ off, i1);
        a + (b - a) * s
    };
    Vec3::new(axis(0x11), axis(0x22), axis(0x33)) * jitter
}

/// The SWIRL displacement (world cells) about the vertical (z) axis: a deterministic
/// rotation of the particle's horizontal offset-from-spawn that grows with age. Pure
/// in `(seed, age, swirl)`. Returns `ZERO` for `swirl == 0`.
#[inline]
fn swirl_offset(seed: u64, age: f32, swirl: f32) -> Vec3 {
    if swirl == 0.0 {
        return Vec3::ZERO;
    }
    // A per-particle phase + a radius that grows slowly with age, rotated by an angle
    // proportional to age·swirl. The seed desyncs neighbouring particles so a column
    // of motes spirals as a body rather than in lockstep.
    let phase = signed_from(seed, 0x5317) * std::f32::consts::PI;
    let angle = phase + age * swirl;
    let radius = 0.12 * age * swirl.abs().min(2.0);
    Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0)
}

/// A particle's FINAL world position: its clean ballistic `pos` plus the
/// deterministic turbulence + swirl offsets. The stored `pos` stays the pure
/// trajectory (so noise never compounds); the offsets are re-derived each splat from
/// the fixed seed + current age, so the path is identical every run.
#[inline]
fn particle_world_pos(pool: &ParticlePool, i: usize) -> Vec3 {
    pool.pos[i]
        + turbulence(pool.seed[i], pool.age[i], pool.jitter[i])
        + swirl_offset(pool.seed[i], pool.age[i], pool.swirl[i])
}

// ───────────────────────────────── tick ─────────────────────────────────────

/// Integrate the pool ONE step and spawn from every emitter. Deterministic given
/// `rng` (same seed + same emitters + same `dt` → identical positions).
///
/// Per frame, for each particle: `pos += vel·dt; vel += accel·dt; age += dt`, and
/// cull `age > lifetime`. Each emitter then spawns `rate·dt` particles (the
/// fractional remainder carried in the pool's per-emitter accumulator so the mean
/// rate is exact), each with `vel + uniform(±vel_spread)`.
///
/// `Screen`-anchored emitters spawn at a sentinel world position; their particles
/// are projected directly to the pinned cell at splat time (see [`splat_particles`]).
///
/// SPEC CONTROLS honoured: `burst` (one-shot), `rate` (continuous), `max_particles`
/// (per-emitter cap), the spawn CONE (`direction`+`spread_rad`+`speed`+`speed_spread`)
/// vs the legacy `vel`/`vel_spread`, `lifetime` ± `lifetime_spread`, and the
/// DETERMINISTIC per-particle seed `hash64(emitter.seed, birth_index)` that drives
/// the reproducible path + the `RandomPick` colour.
pub fn tick_emitters(
    emitters: &[Emitter],
    pool: &mut ParticlePool,
    dt: f32,
    rng: &mut ParticleRng,
) {
    let dt = dt.max(0.0);

    // 0. FLOCKING pre-pass: boid accelerations are computed from
    //    the CURRENT (pre-integration) positions/velocities of an emitter's own
    //    live particles and folded into `pool.vel` BEFORE the general integrate
    //    step below, so a `Flock`-motion particle then falls through the
    //    ordinary `Ballistic` integration (vel/accel → pos) exactly like any
    //    other particle — no separate position-update path needed.
    apply_flocking(emitters, pool, dt);

    // 1. Integrate + cull every live particle. Semi-implicit Euler on the CLEAN
    //    ballistic path; the turbulence/swirl offset is added at splat time (so it
    //    never compounds into the integrated state). Cull decrements the owning
    //    emitter's live count for the `max_particles` cap.
    //
    //    MOTION KINDS: `emitters[owner].motion` selects the per-particle
    //    update. `Ballistic`/`Flock` share the original vel/accel path (Flock's
    //    extra delta-v was already folded into `vel` above); `Curl`/`Attract`
    //    layer a field on top of it; `Lorenz`/`Ripple` REPLACE position with a
    //    closed-form function of the particle's own state/seed/age. An invalid
    //    owner (legacy hand-placed particles, `u16::MAX`) falls back to Ballistic.
    let cap = pool.capacity();
    for i in 0..cap {
        if !pool.alive[i] {
            continue;
        }
        let motion = emitters
            .get(pool.owner[i] as usize)
            .map(|e| e.motion)
            .unwrap_or(MotionKind::Ballistic);
        match motion {
            MotionKind::Ballistic | MotionKind::Flock { .. } => {
                pool.vel[i] += pool.accel[i] * dt;
                pool.pos[i] += pool.vel[i] * dt;
            }
            MotionKind::Curl {
                scale,
                strength,
                evolve,
            } => {
                pool.vel[i] += pool.accel[i] * dt;
                let curl_v =
                    curl2d(pool.seed[i], pool.pos[i], pool.age[i], scale, evolve) * strength;
                pool.pos[i] += (pool.vel[i] + curl_v) * dt;
            }
            MotionKind::Attract {
                target,
                strength,
                swirl,
            } => {
                let to = target - pool.pos[i];
                let d2 = to.length_squared();
                let pull = to * (strength / (d2 + 1.0));
                let tangent = Vec3::new(-to.y, to.x, 0.0).normalize_or_zero() * swirl;
                pool.vel[i] += (pool.accel[i] + pull + tangent) * dt;
                pool.pos[i] += pool.vel[i] * dt;
            }
            MotionKind::Lorenz { scale } => {
                let scale = scale.max(f32::EPSILON);
                let s = pool.lorenz[i];
                let k1 = lorenz_deriv(s);
                let mid = s + k1 * (dt * 0.5);
                let k2 = lorenz_deriv(mid);
                let s_next = s + k2 * dt;
                pool.lorenz[i] = s_next;
                pool.pos[i] = s_next * scale;
            }
            MotionKind::Ripple { speed } => {
                let origin = emitters
                    .get(pool.owner[i] as usize)
                    .map(anchor_world)
                    .unwrap_or(pool.pos[i]);
                let angle = ripple::ring_angle(unit_from(pool.seed[i], 0xF001));
                pool.pos[i] = ripple::ring_pos(origin, angle, pool.age[i], speed);
            }
        }

        // FLOOR BOUNCE (sparks), independent of motion kind: applies
        // whenever the owning emitter authored `bounce` and the particle has sunk
        // to/through its floor while still falling.
        if let Some(em) = emitters.get(pool.owner[i] as usize) {
            if let Some(e) = em.bounce {
                if pool.pos[i].z <= em.floor_z && pool.vel[i].z < 0.0 {
                    pool.pos[i].z = em.floor_z;
                    pool.vel[i].z = -pool.vel[i].z * e;
                    pool.vel[i].x *= SPARK_BOUNCE_FRICTION;
                    pool.vel[i].y *= SPARK_BOUNCE_FRICTION;
                }
            }
            // TRAIL: push the just-computed position into the ring.
            if em.trail.is_some() {
                let mut buf = pool.trail_buf[i];
                for k in (1..TRAIL_CAP).rev() {
                    buf[k] = buf[k - 1];
                }
                buf[0] = pool.pos[i];
                pool.trail_buf[i] = buf;
                pool.trail_n[i] = (pool.trail_n[i] + 1).min(TRAIL_CAP as u8);
            }
        }

        pool.age[i] += dt;
        if pool.age[i] > pool.life[i] {
            pool.alive[i] = false;
            let o = pool.owner[i] as usize;
            if o < pool.live_per_emitter.len() && pool.live_per_emitter[o] > 0 {
                pool.live_per_emitter[o] -= 1;
            }
        }
    }

    // 2. Spawn from each emitter.
    pool.ensure_acc(emitters.len());
    for (ei, em) in emitters.iter().enumerate() {
        // How many to spawn THIS tick: a one-shot burst (once) or the continuous,
        // fractional-accumulated rate.
        let count = if let Some(n) = em.burst {
            if pool.burst_fired[ei] {
                0
            } else {
                pool.burst_fired[ei] = true;
                n as usize
            }
        } else {
            let want = em.rate * dt + pool.spawn_acc[ei];
            let n = want.floor();
            pool.spawn_acc[ei] = want - n;
            n as usize
        };
        if count == 0 {
            continue;
        }

        let base_pos = match em.origin {
            Anchor::World(p) => p,
            Anchor::Screen(c, r) => Vec3::new(c as f32, r as f32, SCREEN_SENTINEL_Z),
        };
        let accel = match em.origin {
            Anchor::World(_) => em.accel,
            Anchor::Screen(_, _) => Vec3::ZERO,
        };
        let uses_cone = em.direction != Vec3::ZERO && em.speed > 0.0;

        for _ in 0..count {
            // Per-emitter cap: stop spawning once the live count hits the ceiling.
            if pool.live_per_emitter[ei] >= em.max_particles {
                break;
            }
            // The DETERMINISTIC per-particle seed (the user requirement): a fixed hash
            // of the emitter seed + this particle's monotonic birth index. Its whole
            // path + colour pick derive from this — never a fresh per-frame roll.
            let birth_index = pool.birth[ei];
            pool.birth[ei] = birth_index.wrapping_add(1);
            let pseed = hash64(em.seed, birth_index);

            // Speed override: `speed_dist` replaces the plain
            // `speed ± speed_spread·randomize` draw when authored.
            let speed_override = em.speed_dist.map(|d| {
                d.sample(unit_from(pseed, 0xD200), unit_from(pseed, 0xD201))
                    .max(0.0)
            });

            let vel = if matches!(em.origin, Anchor::Screen(_, _)) {
                Vec3::ZERO
            } else if uses_cone || speed_override.is_some() {
                cone_velocity(em, pseed, speed_override)
            } else {
                // Legacy demo path: mean vel + per-axis uniform spread (RNG stream).
                em.vel
                    + Vec3::new(
                        em.vel_spread.x * rng.signed(),
                        em.vel_spread.y * rng.signed(),
                        em.vel_spread.z * rng.signed(),
                    )
            };

            // Lifetime ± spread, deterministic from the particle seed. `lifetime_dist`
            // overrides the plain mean±spread when authored; `randomize` scales
            // the spread of the plain path (1.0 = unchanged).
            let life = match em.lifetime_dist {
                Some(d) => d
                    .sample(unit_from(pseed, 0x7100), unit_from(pseed, 0x7101))
                    .max(0.01),
                None => (em.lifetime
                    + em.lifetime_spread * em.randomize * signed_from(pseed, 0x71FE))
                .max(0.01),
            };

            // Birth size: `size_dist` overrides the plain `size` when authored.
            let size0 = match em.size_dist {
                Some(d) => d
                    .sample(unit_from(pseed, 0x5100), unit_from(pseed, 0x5101))
                    .max(f32::EPSILON),
                None => em.size,
            };

            // RandomPick colour: choose one of the three stops at birth, hold it.
            let fg_pick = if em.color_mode == ColorMode::RandomPick {
                let idx = (hash64(pseed, 0xC010) % 3) as usize;
                em.colors[idx]
            } else {
                NO_FG_PICK
            };

            // Per-particle colour jitter: a coherent HSL shift applied to the
            // WHOLE ramp, scaled by `randomize`. `None` ⇒ the shared ramp, unchanged.
            let ramp = match em.color_jitter {
                Some(j) => {
                    let scaled = ColorJitter {
                        h_deg: j.h_deg * em.randomize,
                        s: j.s * em.randomize,
                        l: j.l * em.randomize,
                        dance: j.dance,
                    };
                    em.ramp.jittered(scaled, pseed)
                }
                None => em.ramp,
            };

            // Glyph pick (`glyph_pick = Random`): a fixed member chosen once at
            // birth, held for life. `Sequential` (default) keeps the age-driven pick
            // (`NO_FIXED_GLYPH`).
            let glyph_fixed = if em.glyph_pick == GlyphPick::Random && !em.family.glyphs.is_empty()
            {
                let idx = (hash64(pseed, 0xF100) as usize) % em.family.glyphs.len();
                idx as i16
            } else {
                NO_FIXED_GLYPH
            };

            // Word particles: each spawned particle claims the NEXT character
            // of the authored text, cycling — the word "breaks apart" over the
            // emitter's lifetime as a steady stream of particles carries it away.
            let literal_glyph = match &em.glyph_source {
                GlyphSource::Text(s) => {
                    let chars: Vec<char> = s.chars().collect();
                    if chars.is_empty() {
                        0
                    } else {
                        chars[(birth_index as usize) % chars.len()] as u32
                    }
                }
                GlyphSource::Family => 0,
            };

            // Dust visibility: a birth-time light sample (see `EmitterSpec::
            // visibility` docs for why birth-time, not continuous).
            let vis = match em.visibility {
                DustVisibility::Always => 1.0,
                DustVisibility::Lit { gain } => {
                    (light_at(&em.nearby_lights, base_pos) * gain).clamp(0.0, 1.0)
                }
            };

            // Lorenz raw state seed (`motion = lorenz`): the spawn position IS
            // the initial state (scaled) — see `MotionKind::Lorenz` docs.
            let lorenz0 = if let MotionKind::Lorenz { scale } = em.motion {
                base_pos / scale.max(f32::EPSILON)
            } else {
                Vec3::ZERO
            };

            pool.spawn(SpawnParticle {
                pos: base_pos,
                vel,
                accel,
                life,
                ramp,
                family: em.family,
                seed: pseed,
                size0,
                growth: em.growth,
                jitter: em.path_jitter,
                swirl: em.swirl,
                bold: em.bold,
                fg_pick,
                godray: em.kind == EmitterKind::GodRays,
                ripple_hem: matches!(em.motion, MotionKind::Ripple { .. }),
                owner: ei as u16,
                lorenz0,
                glyph_fixed,
                literal_glyph,
                vis,
            });
            pool.live_per_emitter[ei] += 1;
        }
    }
}

/// A deterministic spawn velocity inside the emitter's CONE: the mean `direction`
/// rotated by a random angle within `spread_rad` (about a perpendicular axis), scaled
/// by `speed ± speed_spread` (or by `speed_override` when `speed_dist` fired).
/// Pure in `(emitter, pseed, speed_override)` — same particle, same initial velocity
/// every run.
#[inline]
fn cone_velocity(em: &Emitter, pseed: u64, speed_override: Option<f32>) -> Vec3 {
    let dir = em.direction; // already unit (from_spec normalised it)
                            // A perpendicular basis to tilt within. Pick any axis not parallel to `dir`.
    let up = if dir.z.abs() < 0.99 { Vec3::Z } else { Vec3::X };
    let tangent = dir.cross(up).normalize_or_zero();
    let bitangent = dir.cross(tangent).normalize_or_zero();
    // Cone angle: sqrt for a uniform-over-the-cap distribution; azimuth full circle.
    let u = (hash64(pseed, 0xA21E) >> 40) as f32 / (1u64 << 24) as f32;
    let theta = em.spread_rad * u.sqrt();
    let azimuth = signed_from(pseed, 0xB33F) * std::f32::consts::PI;
    let (st, ct) = theta.sin_cos();
    let tilt = (tangent * azimuth.cos() + bitangent * azimuth.sin()) * st;
    let cone_dir = (dir * ct + tilt).normalize_or_zero();
    let speed = speed_override.unwrap_or_else(|| {
        (em.speed + em.speed_spread * em.randomize * signed_from(pseed, 0x5EED)).max(0.0)
    });
    cone_dir * speed
}

/// Sentinel `z` marking a screen-anchored particle's `pos` as `(col, row, _)`.
const SCREEN_SENTINEL_Z: f32 = -1_000_000.0;

/// A uniform `f32 ∈ [0,1)` from a particle seed + an integer phase index — the
/// unsigned twin of [`signed_from`], used by [`re_world::Distribution::sample`]
/// callers and the glyph/word/visibility picks. Pure.
#[inline]
fn unit_from(seed: u64, phase: u64) -> f32 {
    (hash64(seed, phase) >> 40) as f32 / (1u64 << 24) as f32
}

/// The world anchor of an [`Emitter`] as a plain [`Vec3`] (a `Screen` anchor
/// returns its `(col, row, SCREEN_SENTINEL_Z)` triple verbatim — motion kinds
/// that need a real world point, like `Ripple`, are only meaningful on a
/// `World` anchor; this never panics on the mismatch, it just degenerates).
#[inline]
fn anchor_world(em: &Emitter) -> Vec3 {
    match em.origin {
        Anchor::World(p) => p,
        Anchor::Screen(c, r) => Vec3::new(c as f32, r as f32, SCREEN_SENTINEL_Z),
    }
}

/// Restitution-independent horizontal damping applied on every floor bounce —
/// sparks lose lateral energy each time they hit the floor, so a burst settles
/// rather than skating forever.
const SPARK_BOUNCE_FRICTION: f32 = 0.92;

/// Smooth (bilinear-interpolated) hash-noise scalar field, sampled at `(x, y)`.
/// Pure in `(seed, x, y)` — the deterministic building block for [`curl2d`].
#[inline]
fn value_noise2(seed: u64, x: f32, y: f32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let cell_hash = |ix: f32, iy: f32| -> f32 {
        let ux = (ix as i64 as u64).wrapping_mul(0xA24B_AED4_963E_E407);
        let uy = (iy as i64 as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25);
        signed_from(seed, ux ^ uy)
    };
    let v00 = cell_hash(xi, yi);
    let v10 = cell_hash(xi + 1.0, yi);
    let v01 = cell_hash(xi, yi + 1.0);
    let v11 = cell_hash(xi + 1.0, yi + 1.0);
    let sx = xf * xf * (3.0 - 2.0 * xf);
    let sy = yf * yf * (3.0 - 2.0 * yf);
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sy
}

/// Divergence-free 2D curl-noise velocity at `pos` (horizontal plane only — see
/// `MotionKind::Curl` docs for why this is a documented simplification of
/// Bridson's 3D curl noise): `v = (∂ψ/∂y, -∂ψ/∂x)` for a single scalar
/// potential `ψ`, evaluated at `pos.xy · scale` with the sample's `y` slid by
/// `age · evolve` (a scrolling potential gives smooth, continuous evolution
/// without a third noise dimension). Pure in `(seed, pos, age, scale, evolve)`.
#[inline]
fn curl2d(seed: u64, pos: Vec3, age: f32, scale: f32, evolve: f32) -> Vec3 {
    const EPS: f32 = 0.05;
    let x = pos.x * scale;
    let y = pos.y * scale + age * evolve;
    let field_seed = seed ^ 0x5A17_C0DE_u64;
    let psi = |xx: f32, yy: f32| value_noise2(field_seed, xx, yy);
    let dpsi_dy = (psi(x, y + EPS) - psi(x, y - EPS)) / (2.0 * EPS);
    let dpsi_dx = (psi(x + EPS, y) - psi(x - EPS, y)) / (2.0 * EPS);
    Vec3::new(dpsi_dy, -dpsi_dx, 0.0)
}

/// One RK2 derivative step of the classic Lorenz attractor (σ=10, ρ=28,
/// β=8/3). Pure — `s` is the raw (unscaled) state.
#[inline]
fn lorenz_deriv(s: Vec3) -> Vec3 {
    const SIGMA: f32 = 10.0;
    const RHO: f32 = 28.0;
    const BETA: f32 = 8.0 / 3.0;
    Vec3::new(
        SIGMA * (s.y - s.x),
        s.x * (RHO - s.z) - s.y,
        s.x * s.y - BETA * s.z,
    )
}

/// The neighbour radius (world cells) within which boids influence each
/// other. Small — fireflies/moths react to close neighbours only.
const FLOCK_NEIGHBOR_RADIUS: f32 = 1.5;

/// Hard speed cap (cells/sec) on the boid-influenced velocity so a flock never
/// runs away numerically (separation can spike when two particles coincide).
const FLOCK_MAX_SPEED: f32 = 3.0;

/// The FLOCKING pre-pass (Reynolds' three rules): for every
/// emitter whose `motion` is `Flock`, gather its own live particles (capped at
/// `n`) and fold a separation + alignment + cohesion + optional
/// light-seeking delta-v into `pool.vel` — read from the CURRENT positions/
/// velocities only (no particle sees another's already-updated state this
/// tick, so the result does not depend on pool iteration order). O(k²) per
/// flock, `k ≤ 64` by the spec's own cap.
fn apply_flocking(emitters: &[Emitter], pool: &mut ParticlePool, dt: f32) {
    if dt <= 0.0 {
        return;
    }
    for (ei, em) in emitters.iter().enumerate() {
        let MotionKind::Flock {
            n,
            sep,
            align,
            coh,
            seek_light,
        } = em.motion
        else {
            continue;
        };
        let cap = pool.capacity();
        let mut members: Vec<usize> = (0..cap)
            .filter(|&i| pool.alive[i] && pool.owner[i] as usize == ei)
            .collect();
        members.truncate(n as usize);
        if members.len() < 2 {
            continue;
        }
        let positions: Vec<Vec3> = members.iter().map(|&i| pool.pos[i]).collect();
        let velocities: Vec<Vec3> = members.iter().map(|&i| pool.vel[i]).collect();
        let nearest_light = em.nearby_lights.first().map(|l| l.pos);

        for (a, &i) in members.iter().enumerate() {
            let mut sep_v = Vec3::ZERO;
            let mut align_v = Vec3::ZERO;
            let mut coh_v = Vec3::ZERO;
            let mut neighbors = 0u32;
            for (b, &pb) in positions.iter().enumerate() {
                if a == b {
                    continue;
                }
                let d = positions[a] - pb;
                let dist = d.length();
                if dist < FLOCK_NEIGHBOR_RADIUS && dist > 1e-4 {
                    sep_v += d / dist / dist; // stronger push the closer they are
                    align_v += velocities[b];
                    coh_v += pb;
                    neighbors += 1;
                }
            }
            if neighbors == 0 {
                continue;
            }
            let inv_n = 1.0 / neighbors as f32;
            align_v = (align_v * inv_n - velocities[a]) * align.max(0.0);
            coh_v = ((coh_v * inv_n) - positions[a]) * coh.max(0.0);
            sep_v *= sep.max(0.0);
            let mut delta_v = sep_v + align_v + coh_v;
            if seek_light > 0.0 {
                if let Some(target) = nearest_light {
                    delta_v += (target - positions[a]).normalize_or_zero() * seek_light;
                }
            }
            let mut v = pool.vel[i] + delta_v * dt;
            let speed = v.length();
            if speed > FLOCK_MAX_SPEED {
                v *= FLOCK_MAX_SPEED / speed;
            }
            pool.vel[i] = v;
        }
    }
}

// ──────────────────────────────── splat ─────────────────────────────────────

/// Project every live STANDARD particle into the emitters layer, depth-tested
/// against `opaque`. Back-compat wrapper for callers that have no god-rays layer:
/// it forwards a throwaway god-rays layer that is discarded. Prefer
/// [`splat_particles_kinds`] (the live path) so god-rays land in their own layer.
///
/// `out` is CLEARED first; only live, visible particles mark cells present, so the
/// layer stays sparse. The glyph comes from the particle's family by normalized age
/// (Growth small→large); the fg from its ramp.
pub fn splat_particles(
    pool: &ParticlePool,
    cam: &CameraPose,
    fov: &Fov,
    opaque: &Layer,
    out: &mut Layer,
) {
    // No god-rays sink supplied: route god-rays particles nowhere (a scratch layer
    // the caller never reads). Existing standard-only callers are byte-identical.
    let (cols, rows) = out.dims();
    let mut scratch = Layer::new(cols, rows, re_core::LayerBlend::ColorMix);
    splat_particles_kinds(pool, cam, fov, opaque, out, &mut scratch);
}

/// Project every live particle into the correct layer by KIND:
/// * [`EmitterKind::Standard`] → `emitters` (a glyph splat, depth-tested vs `opaque`),
/// * [`EmitterKind::GodRays`] → `godrays` (a COLOUR-ONLY, alpha-carrying haze cell,
///   no glyph, NOT depth-tested — a volumetric glow tints whatever it covers).
///
/// Both layers are CLEARED first; only live, visible particles mark cells present, so
/// each layer stays sparse. A standard particle's glyph comes from its family by
/// normalized age and its fg from the ramp; a god-rays particle writes its warm ramp
/// colour into BOTH `fg` and `bg` with an age-and-depth-faded ALPHA in the high byte
/// (the `ColorMix` weight), and glyph `0` (space) — the `ColorMix` resolve keeps the
/// destination glyph regardless.
pub fn splat_particles_kinds(
    pool: &ParticlePool,
    cam: &CameraPose,
    fov: &Fov,
    opaque: &Layer,
    out: &mut Layer,
    godrays: &mut Layer,
) {
    out.clear();
    godrays.clear();
    let (cols, rows) = out.dims();
    if cols == 0 || rows == 0 {
        return;
    }

    let view = view_params(cols, rows, fov);
    let plane_scale = view.plane_scale;
    let proj = view.proj_plane_dist;

    // Camera basis (matches the raycaster's column projection).
    let (dir_x, dir_y) = cam.dir();
    let plane_dir_x = cam.yaw.cos();
    let plane_dir_y = -cam.yaw.sin();
    let horizon = (rows as f32) * 0.5;

    // Project a WORLD point to `(col, row, depth)`, or `None` if it is behind
    // the camera, outside the horizontal FOV, or off the vertical extent.
    // Shared by the main particle pass and the trail-ghost pass below so the
    // two never disagree on where a point lands.
    let project = |p: Vec3| -> Option<(usize, usize, f32)> {
        let rel = p.xy() - cam.pos.xy();
        let depth = rel.x * dir_x + rel.y * dir_y;
        if depth <= 1e-3 {
            return None;
        }
        let lateral = rel.x * plane_dir_x + rel.y * plane_dir_y;
        let camera_x = (lateral / depth) / plane_scale;
        if !(-1.0..=1.0).contains(&camera_x) {
            return None;
        }
        let col = ((camera_x + 1.0) * 0.5 * cols as f32).floor() as i32;
        if col < 0 || col as usize >= cols {
            return None;
        }
        let dz = p.z - cam.pos.z;
        let row = (horizon - proj * dz / depth).floor() as i32;
        if row < 0 || row as usize >= rows {
            return None;
        }
        Some((col as usize, row as usize, depth))
    };

    // Density accumulator: one slot per SCREEN cell, filled by every unoccluded,
    // visible STANDARD particle that lands there this frame — a cluster of
    // overlapping particles resolves to a density-ramp glyph rather than
    // last-write-wins (see module docs).
    let mut accum: Vec<CellAccum> = vec![CellAccum::default(); cols * rows];

    let cap = pool.capacity();
    for i in 0..cap {
        if !pool.alive[i] {
            continue;
        }
        let p = particle_world_pos(pool, i);
        let age = pool.age[i];
        let life = pool.life[i];
        let t = (age / life).clamp(0.0, 1.0);
        let is_godray = pool.godray[i];

        // The warm tint colour (god-rays) or the standard fg, both from the ramp /
        // RandomPick. For god-rays this is the colour the ColorMix layer tints toward.
        let color = if pool.fg_pick[i] != NO_FG_PICK {
            pool.fg_pick[i]
        } else {
            pool.ramp[i].sample(t)
        };

        // Dust visibility: a particle below the visibility floor contributes
        // NOTHING (no glyph, no accumulation weight) — `vis` defaults to 1.0 for
        // every non-`Lit` emitter, so this is a no-op for every existing level.
        let weight = pool.vis[i];
        if !is_godray && weight < DUST_VISIBILITY_FLOOR {
            continue;
        }

        // Screen-anchored particle: pinned to its cell, never occluded, never
        // accumulated (there is exactly one candidate for that cell already).
        if (pool.pos[i].z - SCREEN_SENTINEL_Z).abs() < 0.5 {
            let col = pool.pos[i].x.round() as i32;
            let row = pool.pos[i].y.round() as i32;
            if col >= 0 && row >= 0 && (col as usize) < cols && (row as usize) < rows {
                let (col, row) = (col as usize, row as usize);
                if is_godray {
                    let alpha = godray_alpha(t, 0.0);
                    godrays.set(col, row, godray_cell(color, alpha));
                } else {
                    let glyph = resolve_glyph(pool, i, t);
                    let bold = bold_flag(pool.bold[i], 0.0, t);
                    let hem = pool.ripple_hem[i].then_some(color);
                    out.set(col, row, particle_cell_styled(glyph, color, 0.0, bold, hem));
                }
            }
            continue;
        }

        let Some((col, row, depth)) = project(p) else {
            continue;
        };

        if is_godray {
            // GOD-RAYS: colour-only splat, resolved separately from the
            // standard glyph accumulation below (see module docs).
            let alpha = godray_alpha(t, depth);
            if alpha > 0 {
                let mut cell = godray_cell(color, alpha);
                cell.depth = depth;
                godrays.set(col, row, cell);
            }
            continue;
        }

        // STANDARD OCCLUSION: skip if a solid wall is nearer at this cell.
        if let Some(wall) = opaque.get(col, row) {
            if wall.depth < depth {
                continue;
            }
        }

        let glyph = resolve_glyph(pool, i, t);
        let bold = bold_flag(pool.bold[i], depth, t);
        let acc = &mut accum[row * cols + col];
        acc.n += 1;
        acc.energy += weight;
        acc.color_sum += unpack_rgb_f32(color) * weight;
        if depth < acc.near_depth {
            acc.near_depth = depth;
            acc.near_glyph = glyph;
            acc.near_bold = bold;
            acc.near_hem = pool.ripple_hem[i].then_some(color);
        }
    }

    // Resolve: one glyph per covered cell, either the sole particle's own
    // look (n==1) or a density-ramp glyph over the cluster's averaged colour
    // (n>=2).
    for row in 0..rows {
        for col in 0..cols {
            let acc = &accum[row * cols + col];
            if acc.n == 0 {
                continue;
            }
            let glyph = if acc.n == 1 {
                acc.near_glyph
            } else {
                density_ramp_glyph(acc.n)
            };
            let fg = pack_rgb_f32(acc.color_sum / acc.energy.max(1e-4));
            out.set(
                col,
                row,
                particle_cell_styled(glyph, fg, acc.near_depth, acc.near_bold, acc.near_hem),
            );
        }
    }

    // Trail ghosts: a lightweight SECOND pass, last-write-wins,
    // that only fills cells the accumulation pass left EMPTY — a fading tail
    // never overwrites a real particle glyph. The HEAD's own glyph (already
    // resolved above) is optionally overridden with a direction glyph.
    splat_trails(pool, opaque, out, &project);
}

/// One SCREEN CELL's accumulated standard-particle contributions this frame.
/// `near_depth` starts at `+INFINITY` so the first contributor always "wins"
/// the position; `energy` is the sum of contributing weights
/// (`ParticlePool::vis`, `1.0` unless dust visibility gates it) used both as
/// the density count for the glyph ramp and as the colour-average
/// denominator.
#[derive(Clone, Copy, Debug)]
struct CellAccum {
    n: u32,
    energy: f32,
    color_sum: Vec3,
    near_depth: f32,
    near_glyph: u32,
    near_bold: bool,
    /// The ring colour for a `curly` hem, if the nearest
    /// contributor this cell is a [`MotionKind::Ripple`] particle. Tracked
    /// alongside `near_bold`/`near_glyph` the same way — whichever particle
    /// wins the depth test also decides the hem.
    near_hem: Option<u32>,
}

impl Default for CellAccum {
    fn default() -> Self {
        CellAccum {
            n: 0,
            energy: 0.0,
            color_sum: Vec3::ZERO,
            near_depth: f32::INFINITY,
            near_glyph: u32::from(' '),
            near_bold: false,
            near_hem: None,
        }
    }
}

/// Below this per-particle visibility weight (`ParticlePool::vis`), a dust
/// particle contributes nothing at all — fully dark, not just dim.
const DUST_VISIBILITY_FLOOR: f32 = 0.04;

/// `0x00RRGGBB` → linear-ish `[0,255]³` as a [`Vec3`] (no gamma correction —
/// this mirrors the existing byte-domain colour ramps, not a physically linear
/// average; adequate for a density blend of a few particle colours).
#[inline]
fn unpack_rgb_f32(rgb: u32) -> Vec3 {
    Vec3::new(
        ((rgb >> 16) & 0xFF) as f32,
        ((rgb >> 8) & 0xFF) as f32,
        (rgb & 0xFF) as f32,
    )
}

/// The inverse of [`unpack_rgb_f32`], clamping each channel to a byte.
#[inline]
fn pack_rgb_f32(v: Vec3) -> u32 {
    let r = v.x.round().clamp(0.0, 255.0) as u32;
    let g = v.y.round().clamp(0.0, 255.0) as u32;
    let b = v.z.round().clamp(0.0, 255.0) as u32;
    (r << 16) | (g << 8) | b
}

/// The density-ramp glyph for `n` OVERLAPPING standard particles in one cell:
/// `'·' → ':' → '*' → '%'` for dense clouds. Called only for `n >= 2` (a
/// single occupant keeps its own glyph).
#[inline]
fn density_ramp_glyph(n: u32) -> u32 {
    match n {
        0..=2 => u32::from('·'),
        3..=4 => u32::from(':'),
        5..=7 => u32::from('*'),
        _ => u32::from('%'),
    }
}

/// A particle's glyph, honouring the overrides in priority order:
/// `literal_glyph` (word particles) > `glyph_fixed` (`glyph_pick = Random`)
/// > the age-driven family pick (the default path).
#[inline]
fn resolve_glyph(pool: &ParticlePool, i: usize, t: f32) -> u32 {
    if pool.literal_glyph[i] != 0 {
        return pool.literal_glyph[i];
    }
    if pool.glyph_fixed[i] != NO_FIXED_GLYPH {
        let idx = pool.glyph_fixed[i] as usize;
        if let Some(&g) = pool.family[i].glyphs.get(idx) {
            return g as u32;
        }
    }
    pool.family[i].pick(glyph_drive(pool, i, t))
}

/// Splat the fading TAIL behind every trailed, alive particle, and — when
/// `dir_glyphs` is on — override the head's glyph with a direction glyph
/// derived from its screen-space motion. A second, deliberately simple pass:
/// ghosts are last-write-wins and only drawn into cells the main density
/// accumulation left ABSENT, so a faint tail can never erase a real particle.
fn splat_trails(
    pool: &ParticlePool,
    opaque: &Layer,
    out: &mut Layer,
    project: &dyn Fn(Vec3) -> Option<(usize, usize, f32)>,
) {
    let cap = pool.capacity();
    for i in 0..cap {
        if !pool.alive[i] || pool.trail_n[i] < 2 {
            continue;
        }
        let Some((head_col, head_row, head_depth)) = project(pool.pos[i]) else {
            continue;
        };
        let head_present = out.is_present(head_col, head_row);

        // Direction glyph: compare the head's screen cell to the previous
        // trail point's screen cell.
        if head_present {
            if let Some((pc, pr, _)) = project(pool.trail_buf[i][1]) {
                let dc = head_col as i32 - pc as i32;
                let dr = head_row as i32 - pr as i32;
                if dc != 0 || dr != 0 {
                    let dir_glyph = if dc.abs() >= 2 * dr.abs() {
                        '─'
                    } else if dr.abs() >= 2 * dc.abs() {
                        '│'
                    } else if (dc > 0) == (dr > 0) {
                        '╲'
                    } else {
                        '╱'
                    };
                    if let Some(existing) = out.cell_raw(head_col, head_row) {
                        let mut cell = *existing;
                        cell.glyph = u32::from(dir_glyph);
                        out.set(head_col, head_row, cell);
                    }
                }
            }
        }

        let age = pool.age[i];
        let life = pool.life[i];
        let t = (age / life).clamp(0.0, 1.0);
        let color = if pool.fg_pick[i] != NO_FG_PICK {
            pool.fg_pick[i]
        } else {
            pool.ramp[i].sample(t)
        };
        let n_ghosts = (pool.trail_n[i] as usize).min(TRAIL_CAP);
        for k in 1..n_ghosts {
            let Some((col, row, depth)) = project(pool.trail_buf[i][k]) else {
                continue;
            };
            if out.is_present(col, row) {
                continue; // never overwrite a real particle.
            }
            if let Some(wall) = opaque.get(col, row) {
                if wall.depth < depth {
                    continue;
                }
            }
            // Fade brightness by tail position; fixed FADE_FLOOR default when the
            // owning emitter's `fade` is not threaded through (trail ghosts are
            // pool-only data — no emitter back-reference at splat time, matching
            // the rest of this module's per-particle design). A flat 55% step per
            // tail position is a reasonable default fade curve.
            let fade = TRAIL_GHOST_FADE.powi(k as i32);
            let dim = Vec3::splat(fade);
            let faded = pack_rgb_f32(unpack_rgb_f32(color) * dim);
            let ghost_glyph = if k + 1 == n_ghosts { '.' } else { '·' };
            out.set(
                col,
                row,
                particle_cell_styled(u32::from(ghost_glyph), faded, depth, false, None),
            );
        }
        let _ = head_depth;
    }
}

/// Per-tail-step brightness multiplier for a trail ghost ("tail fade").
/// Applied as `TRAIL_GHOST_FADE.powi(k)`, so each step back is dimmer than
/// the last. A fixed, documented default — see `splat_trails`.
const TRAIL_GHOST_FADE: f32 = 0.6;

/// The god-rays ALPHA (0..=255) packed into the high byte of a shaft cell's
/// `fg`/`bg` — the `ColorMix` lerp weight. The shaft is BRIGHTEST near birth/source
/// and fades to nothing by death (age fade), and DIMMER the farther down-beam it is
/// (depth fade), so a light shaft reads strong at its mouth and dissolves into the
/// distance. Pure; no transcendentals (a couple of muls per particle).
///
/// `t` = normalized age `[0,1]`; `depth` = perpendicular distance (world cells).
#[inline]
fn godray_alpha(t: f32, depth: f32) -> u8 {
    // AGE fade: a soft rise over the first ~15% (no hard pop at birth) then a linear
    // fall to 0 at death — `(1 - t)` shaped, scaled by the peak strength.
    let age_fade = (1.0 - t) * (t / GODRAY_FADE_IN).min(1.0);
    // DEPTH fade: 1 at the camera, linearly to 0 at GODRAY_DEPTH_FADE_CELLS.
    let depth_fade = (1.0 - depth / GODRAY_DEPTH_FADE_CELLS).clamp(0.0, 1.0);
    let a = GODRAY_PEAK_ALPHA * age_fade * depth_fade;
    a.round().clamp(0.0, 255.0) as u8
}

/// Build a colour-only GOD-RAYS [`Cell`]: the warm tint `color` (`0x00RRGGBB`)
/// packed into BOTH `fg` and `bg` with `alpha` in the high byte (`0xAARRGGBB`), glyph
/// `0` (space) and no style. The `godrays` layer's [`re_core::LayerBlend::ColorMix`]
/// reads that alpha as the lerp weight and KEEPS the destination glyph/style/depth,
/// so the shaft tints without erasing any detail.
#[inline]
fn godray_cell(color: u32, alpha: u8) -> Cell {
    let packed = ((alpha as u32) << 24) | (color & 0x00FF_FFFF);
    Cell {
        glyph: 0,
        fg: packed,
        bg: packed,
        depth: 0.0,
        style: CellStyle::NONE,
        ..Cell::default()
    }
}

/// Peak god-rays alpha (the brightest a shaft cell tints). 0..=255; ~70% so a shaft
/// reads as strong warm haze without ever fully overwriting the scene colour.
const GODRAY_PEAK_ALPHA: f32 = 178.0;

/// Fraction of life over which a god-rays particle fades IN (soft mouth, no pop).
const GODRAY_FADE_IN: f32 = 0.15;

/// Perpendicular distance (world cells) at which a god-rays shaft has faded to 0 —
/// the beam dissolves into the distance rather than tinting the far backdrop.
const GODRAY_DEPTH_FADE_CELLS: f32 = 14.0;

/// The age-quantized GROWTH driver for a particle's glyph: effective size over age
/// `lerp(size0, size0·growth, t)` normalised into `[0,1]`, then HELD at a coarse age
/// bucket so the picked family member changes only at a bucket boundary — never every
/// frame (the GLYPH HOLD requirement; mirrors the shader's anti-flicker deadband).
#[inline]
fn glyph_drive(pool: &ParticlePool, i: usize, t: f32) -> f32 {
    // Quantize age to a coarse bucket (≈8 steps over the particle's life) keyed off
    // the particle seed so neighbours desync — the same trick `cell_phase` uses.
    const BUCKETS: f32 = 8.0;
    let bucket = (t * BUCKETS).floor() / BUCKETS;
    let eff = pool.size0[i] * (1.0 + (pool.growth[i] - 1.0) * bucket);
    // Map effective size to [0,1]: size 1 → 0 (smallest member), larger → toward 1.
    // A simple saturating map keeps a size-1 default at the family's small end.
    ((eff - 1.0) * 0.5 + bucket).clamp(0.0, 1.0)
}

/// Resolve a [`BoldMode`] to a bold/normal flag at the given depth + normalized age.
#[inline]
fn bold_flag(mode: BoldMode, depth: f32, t: f32) -> bool {
    match mode {
        BoldMode::Off => false,
        BoldMode::On => true,
        // Bold while close to the camera (small depth). Threshold in world cells.
        BoldMode::ByProximity => depth <= BOLD_PROXIMITY_CELLS,
        // Bold while young (birth → ~20% of life), then normal — a spawn flash.
        BoldMode::ByAge => t <= 0.2,
    }
}

/// Within this perpendicular depth (world cells) a `ByProximity` particle is bold.
const BOLD_PROXIMITY_CELLS: f32 = 4.0;

/// Build the emitter [`Cell`] for a splat: glyph + fg + depth + optional bold
/// style, plus an optional `hem` — a curly colored-underline in the ring's own
/// colour marking a ripple's leading edge (`U(curly ring_col)`). bg
/// transparent: the `GlyphOnly` blend keeps the bg from below at resolve time
/// while also carrying a non-empty `style` through, which is what makes `hem`
/// reach the composited frame at all; see `re_core::layer::apply_emitter_blend`.
#[inline]
fn particle_cell_styled(glyph: u32, fg: u32, depth: f32, bold: bool, hem: Option<u32>) -> Cell {
    let mut style = if bold {
        CellStyle {
            flags: CellStyle::BOLD,
            ..CellStyle::NONE
        }
    } else {
        CellStyle::NONE
    };
    if let Some(ring_col) = hem {
        style.underline = ring_col & 0x00FF_FFFF;
        style.underline_style = UnderlineStyle::Curly;
    }
    Cell {
        glyph,
        fg,
        bg: 0,
        depth,
        style,
        ..Cell::default()
    }
}

// ────────────────────────────── demo emitters ───────────────────────────────

/// The two demo emitters used by a demo level so the system is visibly REAL: a
/// rising ember column and a drifting haze, placed near the test-room pillar.
/// `re_game` can spawn these (or author its own) into the [`EmitterSet`].
pub fn demo_emitters() -> Vec<Emitter> {
    vec![
        Emitter::ember_column(Vec3::new(3.5, 3.0, 0.6)),
        Emitter::drifting_haze(Vec3::new(2.5, 3.0, 1.4)),
    ]
}

// ─────────────────────────────── Bevy plugin ────────────────────────────────

/// The authored emitter sources for the scene. INSERT-IF-ABSENT by the plugin
/// (default = the two demo emitters) so a level can override before adding the
/// plugin. `re_game` replaces this from level data.
#[derive(Resource, Clone, Debug)]
pub struct EmitterSet {
    /// The authored emitters currently live in the scene.
    pub emitters: Vec<Emitter>,
}

impl Default for EmitterSet {
    fn default() -> Self {
        EmitterSet {
            emitters: demo_emitters(),
        }
    }
}

/// The deterministic RNG STREAM resource — its frame index advances each tick so
/// the particle field varies frame-to-frame yet is reproducible from the fixed
/// base seed (golden-frame safe). Owned as a resource so the tick is allocation-free.
#[derive(Resource, Clone, Copy, Debug)]
pub struct EmitterRngState {
    /// Fixed base seed; the per-frame stream seed is `base ^ frame`.
    pub base: u64,
    /// Monotonic frame index (advanced by the tick system).
    pub frame: u64,
}

impl Default for EmitterRngState {
    fn default() -> Self {
        EmitterRngState {
            base: 0xC0FF_EE15_600D_5EED,
            frame: 0,
        }
    }
}

impl EmitterRngState {
    /// A fresh deterministic RNG for the current frame, then advance the frame.
    #[inline]
    pub fn rng_for_frame(&mut self) -> ParticleRng {
        let r = ParticleRng::seed(self.base ^ self.frame.wrapping_mul(0x2545_F491_4F6C_DD1D));
        self.frame = self.frame.wrapping_add(1);
        r
    }
}

/// Marks the emitter tick in `PostUpdate` so the composite system can order itself
/// AFTER it (particles are integrated before they are splatted + resolved).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmitterTickSet;

/// Installs the particle resources + the unconditional per-frame tick (spec §3:
/// emitters integrate every frame). The SPLAT into the composite stack is driven by
/// [`crate::composite::CompositeRenderPlugin`] (which runs after this set), so this
/// plugin owns only the pool + its integration.
pub struct EmitterPlugin;

impl Plugin for EmitterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EmitterSet>()
            .init_resource::<ParticlePool>()
            .init_resource::<EmitterRngState>()
            .add_systems(PostUpdate, tick_emitters_sys.in_set(EmitterTickSet));
    }
}

/// Bevy wrapper: integrate the pool one frame from the live `dt`. Unconditional —
/// the particle motion must stay smooth even when the near geometry is static.
fn tick_emitters_sys(
    set: Res<EmitterSet>,
    time: Res<Time>,
    mut pool: ResMut<ParticlePool>,
    mut rng_state: ResMut<EmitterRngState>,
) {
    let mut rng = rng_state.rng_for_frame();
    tick_emitters(&set.emitters, &mut pool, time.delta_secs(), &mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A legacy demo-style emitter (uses `vel`/`vel_spread`, not the cone) with all
    /// the new spec-driven fields at neutral defaults — the base test literals spread.
    fn legacy_emitter(origin: Vec3, rate: f32, lifetime: f32, vel: Vec3) -> Emitter {
        Emitter {
            origin: Anchor::World(origin),
            rate,
            burst: None,
            lifetime,
            lifetime_spread: 0.0,
            vel,
            vel_spread: Vec3::ZERO,
            direction: Vec3::ZERO,
            spread_rad: 0.0,
            speed: 0.0,
            speed_spread: 0.0,
            accel: Vec3::ZERO,
            size: 1.0,
            growth: 1.0,
            path_jitter: 0.0,
            swirl: 0.0,
            seed: 1,
            bold: BoldMode::Off,
            family: GROWTH_DOT,
            ramp: ColorRamp::two(0xFFFFFF, 0x000000),
            colors: [0xFFFFFF, 0x808080, 0x000000],
            color_mode: ColorMode::RampOverAge,
            max_particles: DEFAULT_POOL_CAP as u32,
            blend: re_core::LayerBlend::GlyphOnly,
            kind: EmitterKind::Standard,
            motion: MotionKind::Ballistic,
            lifetime_dist: None,
            speed_dist: None,
            size_dist: None,
            randomize: 1.0,
            color_jitter: None,
            glyph_pick: GlyphPick::Sequential,
            glyph_source: GlyphSource::Family,
            trail: None,
            bounce: None,
            floor_z: origin.z,
            visibility: DustVisibility::Always,
            nearby_lights: Vec::new(),
        }
    }

    fn one_ember() -> Vec<Emitter> {
        vec![legacy_emitter(
            Vec3::new(0.0, 5.0, 0.0),
            100.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )]
    }

    #[test]
    fn pool_ring_reuse_never_grows_capacity() {
        let mut pool = ParticlePool::with_capacity(64);
        let cap0 = pool.capacity();
        let emitters = one_ember();
        let mut rng = ParticleRng::seed(1);
        // Run far longer than a lifetime so particles are continuously culled and
        // their slots reused. Capacity must never change (no realloc).
        for _ in 0..500 {
            tick_emitters(&emitters, &mut pool, 1.0 / 60.0, &mut rng);
            assert_eq!(pool.capacity(), cap0, "pool capacity must be fixed");
            assert!(pool.live_count() <= cap0);
        }
        // With rate 100/s and lifetime 1s, steady-state live count ≈ 100 but the cap
        // is 64, so the pool saturates and stays full — proving the ring caps cleanly.
        assert!(pool.live_count() > 0);
    }

    #[test]
    fn tick_is_deterministic_for_same_seed() {
        let emitters = one_ember();
        let run = || {
            let mut pool = ParticlePool::with_capacity(256);
            let mut rng = ParticleRng::seed(42);
            for _ in 0..30 {
                tick_emitters(&emitters, &mut pool, 1.0 / 30.0, &mut rng);
            }
            // Snapshot live positions in slot order.
            (0..pool.capacity())
                .filter(|&i| pool.alive[i])
                .map(|i| pool.pos[i])
                .collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "same seed must yield identical particle positions");
        assert!(!a.is_empty(), "expected live particles");
    }

    #[test]
    fn integrate_moves_along_velocity_and_culls() {
        // A particle spawned this frame is integrated on SUBSEQUENT frames: with
        // vel +z it rises, and once age exceeds its lifetime it is culled.
        let emitters = vec![legacy_emitter(
            Vec3::ZERO,
            1.0,
            0.1,
            Vec3::new(0.0, 0.0, 2.0),
        )];
        let mut pool = ParticlePool::with_capacity(16);
        let mut rng = ParticleRng::seed(7);
        // Tick 1: rate 1/s, dt 1s → exactly 1 spawn (integration runs before spawn,
        // so the fresh particle is alive at age 0, sitting at its origin).
        tick_emitters(&emitters, &mut pool, 1.0, &mut rng);
        assert_eq!(pool.live_count(), 1, "spawned particle is alive at age 0");
        let z0 = pool.pos[(0..pool.capacity()).find(|&i| pool.alive[i]).unwrap()].z;
        assert_eq!(z0, 0.0, "fresh particle sits at its origin");

        // Tick 2 (no more spawns this step's worth yet): the existing particle is
        // integrated (rises by vel·dt) AND its age (now 1.0) exceeds lifetime 0.1 →
        // culled. Use rate 0 so no new spawn confuses the count.
        let no_spawn = vec![Emitter {
            rate: 0.0,
            ..emitters[0].clone()
        }];
        tick_emitters(&no_spawn, &mut pool, 1.0, &mut rng);
        assert_eq!(pool.live_count(), 0, "particle past its lifetime is culled");
    }

    #[test]
    fn spawn_rate_is_fractional_accumulated() {
        // rate 0.5/s, dt 1s → spawns on alternate ticks (0,1,0,1,…) so over 10 ticks
        // ≈ 5 spawns. Proves the fractional accumulator (no rounding bias).
        let emitters = vec![legacy_emitter(Vec3::ZERO, 0.5, 100.0, Vec3::ZERO)];
        let mut pool = ParticlePool::with_capacity(64);
        let mut rng = ParticleRng::seed(3);
        for _ in 0..10 {
            tick_emitters(&emitters, &mut pool, 1.0, &mut rng);
        }
        assert_eq!(pool.live_count(), 5, "0.5/s over 10s = 5 spawns");
    }

    #[test]
    fn splat_occlusion_skips_particle_behind_wall() {
        // Put a particle straight ahead, then place a near opaque wall in its cell.
        let cols = 40usize;
        let rows = 24usize;
        let fov = Fov::default();
        let cam = CameraPose {
            pos: Vec3::new(0.0, 0.0, 0.5),
            yaw: 0.0, // faces +y (dir = (0,1))
            pitch: 0.0,
        };
        // A pool with one hand-placed particle 5 cells ahead at the camera height.
        let mut pool = ParticlePool::with_capacity(4);
        pool.pos[0] = Vec3::new(0.0, 5.0, 0.5);
        pool.vel[0] = Vec3::ZERO;
        pool.age[0] = 0.0;
        pool.life[0] = 1.0;
        pool.ramp[0] = ColorRamp::two(0xFF00FF, 0);
        pool.family[0] = GROWTH_DOT;
        pool.alive[0] = true;

        // 1. No wall → the particle splats SOMEWHERE.
        let mut opaque = Layer::new(cols, rows, re_core::LayerBlend::Opaque);
        let mut out = Layer::new(cols, rows, re_core::LayerBlend::GlyphOnly);
        splat_particles(&pool, &cam, &fov, &opaque, &mut out);
        let visible = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .any(|(c, r)| out.is_present(c, r));
        assert!(visible, "unoccluded particle must splat");

        // Find the cell it landed in.
        let cell = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .find(|&(c, r)| out.is_present(c, r))
            .unwrap();

        // 2. Now put a NEAR wall (depth 1.0 < particle depth 5.0) at that cell → skip.
        opaque.set(
            cell.0,
            cell.1,
            Cell {
                depth: 1.0,
                ..Cell::default()
            },
        );
        splat_particles(&pool, &cam, &fov, &opaque, &mut out);
        assert!(
            !out.is_present(cell.0, cell.1),
            "particle behind a nearer wall must be occluded"
        );
    }

    #[test]
    fn color_ramp_interpolates() {
        let r = ColorRamp::two(0x000000, 0xFFFFFF);
        assert_eq!(r.sample(0.0), 0x000000);
        assert_eq!(r.sample(1.0), 0xFFFFFF);
        let mid = r.sample(0.5);
        // Mid grey, each channel ≈ 0x7F or 0x80.
        let g = (mid >> 8) & 0xFF;
        assert!((0x7E..=0x81).contains(&g), "mid grey, got {:#X}", mid);
    }

    // ── from_spec round-trip ─────────────────────────────────────────────────

    #[test]
    fn from_spec_round_trips_every_field() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(5.0, 6.0, 1.5)))
            .with_colors(0xFFD27F, 0xFF7A1A, 0x6E1200)
            .with_color_mode(ColorMode::RampOverAge)
            .with_family("growth_dot")
            .with_bold(BoldMode::ByAge)
            .with_direction(Vec3::new(0.0, 0.0, 1.0), 30.0)
            .with_size(1.5, 2.0)
            .with_rate(24.0)
            .with_speed(0.9, 0.2)
            .with_path_rng(0.4, 4242, 0.6)
            .with_lifetime(2.2, 0.3)
            .with_accel(Vec3::new(0.0, 0.0, 0.35))
            .with_max_particles(128)
            .with_blend(EmitterBlend::KeepBg)
            .validated();
        let em = Emitter::from_spec(&spec);

        assert_eq!(em.origin, Anchor::World(Vec3::new(5.0, 6.0, 1.5)));
        assert_eq!(em.rate, 24.0);
        assert_eq!(em.lifetime, 2.2);
        assert_eq!(em.lifetime_spread, 0.3);
        assert_eq!(em.speed, 0.9);
        assert_eq!(em.speed_spread, 0.2);
        assert_eq!(em.size, 1.5);
        assert_eq!(em.growth, 2.0);
        assert_eq!(em.path_jitter, 0.4);
        assert_eq!(em.swirl, 0.6);
        assert_eq!(em.seed, 4242);
        assert_eq!(em.bold, BoldMode::ByAge);
        assert_eq!(em.max_particles, 128);
        assert_eq!(em.accel, Vec3::new(0.0, 0.0, 0.35));
        assert_eq!(em.colors, [0xFFD27F, 0xFF7A1A, 0x6E1200]);
        // Direction is normalised; (0,0,1) stays (0,0,1).
        assert!((em.direction - Vec3::Z).length() < 1e-5);
        // spread_deg 30° → radians.
        assert!((em.spread_rad - 30.0_f32.to_radians()).abs() < 1e-5);
        assert_eq!(em.family.name, "growth_dot");
        assert_eq!(em.blend, re_core::LayerBlend::GlyphOnly);
    }

    #[test]
    fn from_spec_unknown_family_falls_back() {
        let spec = EmitterSpec::new().with_family("no_such_family");
        let em = Emitter::from_spec(&spec);
        assert_eq!(
            em.family.name, GROWTH_DOT.name,
            "unknown family → GROWTH_DOT"
        );
    }

    #[test]
    fn from_spec_glyph_set_overrides_family() {
        let spec = EmitterSpec::new().with_glyph_set(vec!['.', 'o', 'O', '@']);
        let em = Emitter::from_spec(&spec);
        assert_eq!(em.family.glyphs, &['.', 'o', 'O', '@']);
    }

    #[test]
    fn from_spec_screen_origin_maps_to_anchor_screen() {
        let spec = EmitterSpec::new().with_origin(EmitterOrigin::Screen {
            col: 12.0,
            row: 4.0,
        });
        let em = Emitter::from_spec(&spec);
        assert_eq!(em.origin, Anchor::Screen(12, 4));
    }

    // ── deterministic, frame-stable path randomisation ───────────────────────

    fn jitter_emitter(seed: u64) -> Vec<Emitter> {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(0.0, 5.0, 0.0)))
            .with_direction(Vec3::Z, 0.0)
            .with_speed(1.0, 0.0)
            .with_rate(40.0)
            .with_lifetime(2.0, 0.0)
            .with_path_rng(0.8, seed, 0.5)
            .with_accel(Vec3::ZERO)
            .validated();
        vec![Emitter::from_spec(&spec)]
    }

    /// Snapshot the FINAL (turbulence+swirl-displaced) world positions of a run.
    fn run_paths(seed: u64) -> Vec<Vec3> {
        let emitters = jitter_emitter(seed);
        let mut pool = ParticlePool::with_capacity(512);
        let mut rng = ParticleRng::seed(seed);
        for _ in 0..40 {
            tick_emitters(&emitters, &mut pool, 1.0 / 30.0, &mut rng);
        }
        (0..pool.capacity())
            .filter(|&i| pool.alive[i])
            .map(|i| particle_world_pos(&pool, i))
            .collect()
    }

    #[test]
    fn path_is_deterministic_across_runs() {
        // Same seed → identical jittered/swirled paths, every run (user requirement).
        let a = run_paths(98765);
        let b = run_paths(98765);
        assert_eq!(a, b, "same seed+index must trace identical paths");
        assert!(!a.is_empty(), "expected live particles");
    }

    #[test]
    fn path_jitter_actually_displaces() {
        // With jitter on, the final world position differs from the clean ballistic
        // pos — i.e. the turbulence is real, not a no-op.
        let emitters = jitter_emitter(135);
        let mut pool = ParticlePool::with_capacity(256);
        let mut rng = ParticleRng::seed(135);
        for _ in 0..30 {
            tick_emitters(&emitters, &mut pool, 1.0 / 30.0, &mut rng);
        }
        let mut any_offset = false;
        for i in 0..pool.capacity() {
            if pool.alive[i] && pool.age[i] > 0.3 {
                let off = (particle_world_pos(&pool, i) - pool.pos[i]).length();
                if off > 1e-4 {
                    any_offset = true;
                    break;
                }
            }
        }
        assert!(any_offset, "path_jitter/swirl must displace particles");
    }

    #[test]
    fn path_offset_is_stable_at_fixed_age() {
        // The turbulence is a function of (seed, age) ONLY — so for a fixed seed it
        // gives the SAME offset at the SAME age every call (no frame re-roll).
        let off1 = turbulence(0xABCDEF, 0.73, 0.8) + swirl_offset(0xABCDEF, 0.73, 0.5);
        let off2 = turbulence(0xABCDEF, 0.73, 0.8) + swirl_offset(0xABCDEF, 0.73, 0.5);
        assert_eq!(off1, off2, "same (seed,age) must give the same path offset");
    }

    // ── glyph hold (no per-frame churn at a constant age bucket) ──────────────

    #[test]
    fn particle_glyph_does_not_flip_every_frame() {
        // Step a single particle in tiny dt increments WITHIN one age bucket and
        // assert its picked glyph never changes — the GLYPH HOLD requirement.
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(0.0, 5.0, 0.0)))
            .with_direction(Vec3::Z, 0.0)
            .with_speed(0.0, 0.0)
            .with_size(1.0, 3.0) // grows, so the glyph WOULD move if not held
            .with_rate(1.0)
            .with_lifetime(8.0, 0.0)
            .validated();
        let emitters = vec![Emitter::from_spec(&spec)];
        let mut pool = ParticlePool::with_capacity(8);
        let mut rng = ParticleRng::seed(5);
        // Spawn one.
        tick_emitters(&emitters, &mut pool, 1.0, &mut rng);
        let pi = (0..pool.capacity()).find(|&i| pool.alive[i]).unwrap();
        let stop = vec![Emitter {
            rate: 0.0,
            ..emitters[0].clone()
        }];

        // Drive small steps that stay inside one 1/8-of-life bucket (life 8 → bucket
        // width 1s). Advance from age 1.0 to ~1.9 in 0.05 steps: same bucket.
        let glyph0 = pool.family[pi].pick(glyph_drive(
            &pool,
            pi,
            (pool.age[pi] / pool.life[pi]).clamp(0.0, 1.0),
        ));
        for _ in 0..16 {
            tick_emitters(&stop, &mut pool, 0.05, &mut rng);
            if !pool.alive[pi] {
                break;
            }
            let t = (pool.age[pi] / pool.life[pi]).clamp(0.0, 1.0);
            // Only assert while still in the first bucket (age < 1.0 of an 8 s life).
            if (t * 8.0).floor() == 1.0 {
                let g = pool.family[pi].pick(glyph_drive(&pool, pi, t));
                assert_eq!(g, glyph0, "glyph must HOLD within one age bucket");
            }
        }
    }

    // ── burst + max_particles ────────────────────────────────────────────────

    #[test]
    fn burst_emits_once_then_stops() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::ZERO))
            .with_burst(20)
            .with_lifetime(100.0, 0.0)
            .with_speed(0.0, 0.0)
            .validated();
        let emitters = vec![Emitter::from_spec(&spec)];
        let mut pool = ParticlePool::with_capacity(128);
        let mut rng = ParticleRng::seed(9);
        tick_emitters(&emitters, &mut pool, 1.0 / 30.0, &mut rng);
        assert_eq!(pool.live_count(), 20, "burst emits exactly its count");
        // Subsequent ticks add NOTHING (one-shot).
        for _ in 0..10 {
            tick_emitters(&emitters, &mut pool, 1.0 / 30.0, &mut rng);
        }
        assert_eq!(pool.live_count(), 20, "burst must not re-fire");
    }

    #[test]
    fn max_particles_caps_live_count() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::ZERO))
            .with_rate(1000.0)
            .with_lifetime(100.0, 0.0)
            .with_speed(0.0, 0.0)
            .with_max_particles(7)
            .validated();
        let emitters = vec![Emitter::from_spec(&spec)];
        let mut pool = ParticlePool::with_capacity(1024);
        let mut rng = ParticleRng::seed(11);
        for _ in 0..10 {
            tick_emitters(&emitters, &mut pool, 1.0, &mut rng);
        }
        assert_eq!(
            pool.live_count(),
            7,
            "max_particles must cap the live count"
        );
    }

    #[test]
    fn cone_velocity_respects_direction_and_speed() {
        // A pencil-thin cone (spread 0) along +z at speed 2 → velocity ≈ (0,0,2).
        let spec = EmitterSpec::new()
            .with_direction(Vec3::Z, 0.0)
            .with_speed(2.0, 0.0)
            .validated();
        let em = Emitter::from_spec(&spec);
        let v = cone_velocity(&em, 12345, None);
        assert!(
            (v - Vec3::new(0.0, 0.0, 2.0)).length() < 1e-4,
            "pencil cone → +z·speed, got {v:?}"
        );
    }

    // ── shipped recipes: strong per-particle randomisation (complaint B) ──────

    /// Spawn two particles from a shipped recipe and assert their initial velocity
    /// DIRECTIONS diverge by a meaningful angle — proving the widened spread makes
    /// neighbouring particles visibly differ instead of marching in a tight stream.
    /// Determinism is preserved (the spread is seeded), covered by the existing
    /// `tick_is_deterministic_for_same_seed` / `path_is_deterministic_across_runs`.
    fn first_two_velocities(em: Emitter) -> (Vec3, Vec3) {
        let emitters = vec![em];
        let mut pool = ParticlePool::with_capacity(16);
        let mut rng = ParticleRng::seed(0xABCD);
        // dt·rate ≥ 2 guarantees at least two spawns on the first tick. The fresh
        // particles sit at age 0 with their spawn velocity untouched by integration.
        tick_emitters(&emitters, &mut pool, 1.0, &mut rng);
        let live: Vec<usize> = (0..pool.capacity()).filter(|&i| pool.alive[i]).collect();
        assert!(
            live.len() >= 2,
            "need at least two spawned particles, got {}",
            live.len()
        );
        (pool.vel[live[0]], pool.vel[live[1]])
    }

    fn angle_between(a: Vec3, b: Vec3) -> f32 {
        let na = a.normalize_or_zero();
        let nb = b.normalize_or_zero();
        na.dot(nb).clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn ember_column_particles_diverge_in_direction() {
        let (v0, v1) = first_two_velocities(Emitter::ember_column(Vec3::new(3.5, 3.0, 0.6)));
        let ang = angle_between(v0, v1);
        // > ~17° of divergence: the widened vel_spread makes neighbours clearly differ.
        assert!(
            ang > 0.3,
            "ember particles should diverge in direction (got {ang} rad), \
             v0={v0:?} v1={v1:?}"
        );
    }

    #[test]
    fn drifting_haze_particles_diverge_in_direction() {
        let (v0, v1) = first_two_velocities(Emitter::drifting_haze(Vec3::new(2.5, 3.0, 1.4)));
        let ang = angle_between(v0, v1);
        assert!(
            ang > 0.3,
            "haze particles should diverge in direction (got {ang} rad), \
             v0={v0:?} v1={v1:?}"
        );
    }

    #[test]
    fn shipped_recipes_are_calmer_and_more_random() {
        // Lock in the playtest tuning: the shipped recipes are slower (complaint A)
        // and carry real per-particle variation knobs (complaint B).
        let ember = Emitter::ember_column(Vec3::ZERO);
        assert!(ember.vel.z <= 0.5, "ember rise should be halved/calmer");
        assert!(ember.rate <= 16.0, "ember rate trimmed for readability");
        assert!(ember.lifetime_spread > 0.0, "ember needs lifetime variety");
        assert!(
            ember.path_jitter > 0.0 || ember.swirl > 0.0,
            "ember needs path variety"
        );

        let haze = Emitter::drifting_haze(Vec3::ZERO);
        assert!(haze.vel.x <= 0.1, "haze drift should be halved/calmer");
        assert!(haze.rate <= 9.0, "haze rate trimmed for readability");
        assert!(haze.lifetime_spread > 0.0, "haze needs lifetime variety");
        assert!(
            haze.path_jitter > 0.0 || haze.swirl > 0.0,
            "haze needs path variety"
        );
    }

    // ── god-rays: colour-only shafts that tint without erasing glyphs ─────────

    /// A GodRays emitter must route its particles to the `godrays` layer as
    /// COLOUR-ONLY, alpha-carrying cells (no glyph), leaving the `emitters` (glyph)
    /// layer empty — and a `ColorMix` fold of that cell over a wall glyph must TINT
    /// the colour while KEEPING the glyph.
    #[test]
    fn godrays_splat_makes_alpha_color_cells_keeping_glyph() {
        let cols = 60usize;
        let rows = 30usize;
        let fov = Fov::default();
        let cam = CameraPose {
            pos: Vec3::new(0.0, 0.0, 0.5),
            yaw: 0.0, // faces +y
            pitch: 0.0,
        };
        // A GodRays emitter a few cells ahead, beam pointing toward the camera-ish so
        // particles spread across the view.
        let em = Emitter::god_rays(Vec3::new(0.0, 4.0, 0.6), Vec3::new(0.0, -1.0, 0.0));
        let emitters = vec![em];
        let mut pool = ParticlePool::with_capacity(512);
        let mut rng = ParticleRng::seed(0x60D5_BEAD_u64 ^ 0x1234);
        for _ in 0..20 {
            tick_emitters(&emitters, &mut pool, 1.0 / 30.0, &mut rng);
        }
        // No occluder.
        let opaque = Layer::new(cols, rows, re_core::LayerBlend::Opaque);
        let mut glyph_layer = Layer::new(cols, rows, re_core::LayerBlend::GlyphOnly);
        let mut ray_layer = Layer::new(cols, rows, re_core::LayerBlend::ColorMix);
        splat_particles_kinds(&pool, &cam, &fov, &opaque, &mut glyph_layer, &mut ray_layer);

        // The standard glyph layer must be EMPTY (a god-rays emitter draws no glyph).
        let glyph_present = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .any(|(c, r)| glyph_layer.is_present(c, r));
        assert!(
            !glyph_present,
            "GodRays must not write the glyph (emitters) layer"
        );

        // The god-rays layer must carry at least one alpha-bearing colour cell.
        let ray = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .find_map(|(c, r)| ray_layer.get(c, r).map(|cell| (c, r, *cell)));
        let (rc, rr, ray_cell) = ray.expect("GodRays must splat at least one colour cell");
        // glyph 0 (space), alpha packed in the high byte of fg/bg, fg==bg (warm tint).
        assert_eq!(ray_cell.glyph, 0, "god-rays cell carries no glyph");
        let alpha = (ray_cell.fg >> 24) & 0xFF;
        assert!(alpha > 0, "god-rays cell must carry a non-zero alpha");
        assert_eq!(
            ray_cell.fg, ray_cell.bg,
            "fg and bg both carry the warm tint"
        );

        // Now FOLD it over a base wall glyph via ColorMix and prove the tint keeps the
        // glyph. Use the full CompositeStack so we exercise the real resolve.
        let mut stack = re_core::CompositeStack::new(cols, rows);
        let mut base = Cell {
            glyph: u32::from('W'),
            fg: 0x20_20_20,
            bg: 0x05_05_05,
            depth: 6.0,
            ..Cell::default()
        };
        base.style = CellStyle::NONE;
        stack.opaque.set(rc, rr, base);
        stack.godrays.set(rc, rr, ray_cell);
        let mut fb = re_core::Framebuffer::new(cols, rows);
        stack.resolve(&mut fb);
        let out = fb.get(rc, rr).copied().unwrap();
        assert_eq!(
            out.glyph,
            u32::from('W'),
            "the wall glyph must persist under the shaft"
        );
        assert_ne!(out.fg, 0x20_20_20, "the shaft must tint the foreground");
        // The warm tint pushes the red channel UP.
        assert!(
            (out.fg >> 16) & 0xFF >= 0x20,
            "red channel rises toward warm"
        );
    }

    /// A STANDARD emitter is unchanged by the kind plumbing: it still splats into the
    /// glyph (emitters) layer and writes NOTHING into the god-rays layer.
    #[test]
    fn standard_emitter_unaffected_by_godrays_routing() {
        let cols = 40usize;
        let rows = 24usize;
        let fov = Fov::default();
        let cam = CameraPose {
            pos: Vec3::new(0.0, 0.0, 0.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let em = Emitter::ember_column(Vec3::new(0.0, 5.0, 0.5));
        let emitters = vec![em];
        let mut pool = ParticlePool::with_capacity(512);
        let mut rng = ParticleRng::seed(77);
        for _ in 0..30 {
            tick_emitters(&emitters, &mut pool, 1.0 / 30.0, &mut rng);
        }
        let opaque = Layer::new(cols, rows, re_core::LayerBlend::Opaque);
        let mut glyph_layer = Layer::new(cols, rows, re_core::LayerBlend::GlyphOnly);
        let mut ray_layer = Layer::new(cols, rows, re_core::LayerBlend::ColorMix);
        splat_particles_kinds(&pool, &cam, &fov, &opaque, &mut glyph_layer, &mut ray_layer);

        let glyph_present = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .any(|(c, r)| glyph_layer.is_present(c, r));
        assert!(
            glyph_present,
            "a standard emitter must splat into the glyph layer"
        );
        let ray_present = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .any(|(c, r)| ray_layer.is_present(c, r));
        assert!(
            !ray_present,
            "a standard emitter must NOT touch the god-rays layer"
        );
    }

    #[test]
    fn godray_alpha_fades_with_age_and_depth() {
        // Bright near birth (small t) and near the camera (small depth); zero at death
        // (t=1) and beyond the depth-fade reach.
        let near_young = godray_alpha(0.2, 0.0);
        assert!(near_young > 0, "a young near shaft must tint");
        assert_eq!(
            godray_alpha(1.0, 0.0),
            0,
            "at death the shaft is fully faded"
        );
        assert_eq!(
            godray_alpha(0.2, GODRAY_DEPTH_FADE_CELLS + 1.0),
            0,
            "beyond the depth-fade reach the shaft is gone"
        );
        // Farther down-beam is dimmer than near the source at the same age.
        let far = godray_alpha(0.2, GODRAY_DEPTH_FADE_CELLS * 0.5);
        assert!(far < near_young, "down-beam must be dimmer than the mouth");
    }

    #[test]
    fn from_spec_carries_godrays_kind() {
        let spec = EmitterSpec::new().with_kind(EmitterKind::GodRays);
        let em = Emitter::from_spec(&spec);
        assert_eq!(
            em.kind,
            EmitterKind::GodRays,
            "from_spec must carry the kind"
        );
        let std = Emitter::from_spec(&EmitterSpec::new());
        assert_eq!(std.kind, EmitterKind::Standard, "default kind is Standard");
    }

    #[test]
    fn bold_modes_resolve() {
        assert!(!bold_flag(BoldMode::Off, 0.0, 0.0));
        assert!(bold_flag(BoldMode::On, 100.0, 1.0));
        assert!(bold_flag(BoldMode::ByProximity, 1.0, 0.5));
        assert!(!bold_flag(BoldMode::ByProximity, 100.0, 0.5));
        assert!(bold_flag(BoldMode::ByAge, 5.0, 0.1));
        assert!(!bold_flag(BoldMode::ByAge, 5.0, 0.9));
    }

    // ── motion kinds & variety/randomisation ────────────────────────────────
    // Each test below drives the SAME public path a shipped level uses
    // (`EmitterSpec` → `Emitter::from_spec[_lit]` → `tick_emitters` →, where
    // relevant, `splat_particles_kinds`) rather than poking `Emitter` fields
    // directly, so a regression here is a regression a level author would hit.

    fn run_ticks(emitters: &[Emitter], cap: usize, seed: u64, dt: f32, n: usize) -> ParticlePool {
        let mut pool = ParticlePool::with_capacity(cap);
        let mut rng = ParticleRng::seed(seed);
        for _ in 0..n {
            tick_emitters(emitters, &mut pool, dt, &mut rng);
        }
        pool
    }

    #[test]
    fn curl_motion_is_deterministic_and_diverges_from_ballistic() {
        let curl_spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(0.0, 0.0, 1.0)))
            .with_rate(20.0)
            .with_lifetime(2.0, 0.0)
            .with_motion(MotionKind::Curl {
                scale: 0.6,
                strength: 1.5,
                evolve: 0.2,
            });
        let em = Emitter::from_spec(&curl_spec);

        let run = || {
            let pool = run_ticks(std::slice::from_ref(&em), 64, 11, 1.0 / 30.0, 20);
            (0..pool.capacity())
                .filter(|&i| pool.alive[i])
                .map(|i| pool.pos[i])
                .collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "same seed must yield identical curl trajectories");
        assert!(!a.is_empty(), "expected live curl particles");

        // A ballistic emitter with the same spawn parameters (zero accel/vel) would
        // leave every particle sitting at its birth column; curl must move at least
        // one particle noticeably off its spawn X/Y.
        let moved = a.iter().any(|p| (p.x.abs() + p.y.abs()) > 0.05);
        assert!(
            moved,
            "curl motion must displace particles off the spawn point"
        );
    }

    #[test]
    fn attract_motion_pulls_particles_toward_the_target() {
        let target = Vec3::new(10.0, 0.0, 1.0);
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(0.0, 0.0, 1.0)))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(5.0, 0.0)
            .with_motion(MotionKind::Attract {
                target,
                strength: 8.0,
                swirl: 0.0,
            });
        let em = Emitter::from_spec(&spec);
        let pool = run_ticks(&[em], 4, 5, 1.0 / 30.0, 60);
        let i = (0..pool.capacity())
            .find(|&i| pool.alive[i])
            .expect("one live particle");
        let d0 = (target - Vec3::new(0.0, 0.0, 1.0)).length();
        let d1 = (target - pool.pos[i]).length();
        assert!(
            d1 < d0,
            "attract motion must reduce distance to the target ({d0} -> {d1})"
        );
    }

    #[test]
    fn lorenz_motion_is_deterministic_and_bounded() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(2.0, 2.0, 2.0)))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(10.0, 0.0)
            .with_motion(MotionKind::Lorenz { scale: 0.08 });
        let em = Emitter::from_spec(&spec);

        let run = || {
            let pool = run_ticks(std::slice::from_ref(&em), 4, 3, 1.0 / 60.0, 120);
            let i = (0..pool.capacity()).find(|&i| pool.alive[i]).unwrap();
            pool.pos[i]
        };
        let p_a = run();
        let p_b = run();
        assert_eq!(p_a, p_b, "same seed must reproduce the same Lorenz state");
        assert!(
            p_a.is_finite(),
            "Lorenz integration must not blow up to inf/NaN"
        );
    }

    #[test]
    fn ripple_motion_matches_the_ring_formula() {
        let origin = Vec3::new(3.0, 4.0, 0.5);
        let speed = 2.0;
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(origin))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(10.0, 0.0)
            .with_motion(MotionKind::Ripple { speed });
        let em = Emitter::from_spec(&spec);
        let dt = 1.0 / 30.0;
        let pool = run_ticks(&[em], 4, 9, dt, 30);
        let i = (0..pool.capacity())
            .find(|&i| pool.alive[i])
            .expect("one live ripple particle");
        // The ring stays on the origin's horizontal plane and its radius from
        // origin grows with age at exactly `speed` cells/sec. The particle's
        // position is (correctly) computed from the age BEFORE this tick's final
        // increment — the same one-tick lag every motion kind's integrate step
        // has — so compare against `age - dt`, not the post-increment age.
        assert!(
            (pool.pos[i].z - origin.z).abs() < 1e-4,
            "a ripple stays on its plane"
        );
        let r = (pool.pos[i] - Vec3::new(origin.x, origin.y, pool.pos[i].z)).length();
        let expected_r = speed * (pool.age[i] - dt).max(0.0);
        assert!(
            (r - expected_r).abs() < 1e-3,
            "ring radius must equal speed*age (got {r}, want {expected_r})"
        );
    }

    /// A ripple particle's splat cell must carry a `curly` colored underline
    /// in the ring's own colour (`U(curly ring_col)`), and that style must
    /// survive the REAL `CompositeStack` fold (not just the raw layer cell) —
    /// i.e. `re_core::layer::apply_emitter_blend`'s `GlyphOnly` arm must not
    /// silently drop it.
    #[test]
    fn ripple_particle_hem_survives_the_composite_fold() {
        let cols = 20usize;
        let rows = 12usize;
        let cam = CameraPose {
            pos: Vec3::new(0.0, 0.0, 0.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let fov = Fov::default();

        let mut pool = ParticlePool::with_capacity(4);
        pool.pos[0] = Vec3::new(0.0, 5.0, 0.5);
        pool.vel[0] = Vec3::ZERO;
        pool.age[0] = 0.5;
        pool.life[0] = 2.0;
        pool.ramp[0] = ColorRamp::two(0x40A0FF, 0x40A0FF);
        pool.family[0] = GROWTH_DOT;
        pool.ripple_hem[0] = true;
        pool.alive[0] = true;

        let opaque = Layer::new(cols, rows, re_core::LayerBlend::Opaque);
        let mut glyph_layer = Layer::new(cols, rows, re_core::LayerBlend::GlyphOnly);
        let mut ray_layer = Layer::new(cols, rows, re_core::LayerBlend::ColorMix);
        splat_particles_kinds(&pool, &cam, &fov, &opaque, &mut glyph_layer, &mut ray_layer);

        let (col, row, src_cell) = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .find_map(|(c, r)| glyph_layer.get(c, r).map(|cell| (c, r, *cell)))
            .expect("the ripple particle must splat somewhere");
        assert!(
            src_cell.style.has_underline_color(),
            "the splat cell itself must carry a hem"
        );
        assert_eq!(src_cell.style.underline_style(), UnderlineStyle::Curly);

        // Fold it over a bare floor cell via the REAL CompositeStack and confirm the
        // hem — not just the glyph/fg — reaches the resolved frame.
        let mut stack = re_core::CompositeStack::new(cols, rows);
        stack.opaque.set(
            col,
            row,
            Cell {
                glyph: u32::from('.'),
                fg: 0x10_10_10,
                bg: 0x02_02_02,
                depth: 6.0,
                ..Cell::default()
            },
        );
        stack.emitters.set(col, row, src_cell);
        let mut fb = re_core::Framebuffer::new(cols, rows);
        stack.resolve(&mut fb);
        let out = fb.get(col, row).copied().unwrap();
        assert!(
            out.style.has_underline_color(),
            "the resolved frame must keep the ripple's hem"
        );
        assert_eq!(out.style.underline_style(), UnderlineStyle::Curly);
        assert_eq!(
            out.style.underline_color(),
            Some(0x40A0FF),
            "hem colour is the ring's own colour"
        );
        // The bg must still come from BELOW (GlyphOnly keeps bg) — the hem addition
        // must not regress that half of the contract.
        assert_eq!(out.bg, 0x02_02_02, "GlyphOnly must still keep the base bg");
    }

    #[test]
    fn flock_motion_pulls_a_group_closer_together_over_time() {
        // Two burst emitters seeding particles far apart, both owned by a `Flock`
        // motion with cohesion but no seek_light — cohesion should pull members of
        // the SAME flock (same emitter index) toward each other's centroid.
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(0.0, 0.0, 1.0)))
            .with_rate(1.0)
            .with_burst(8)
            .with_lifetime(10.0, 0.0)
            .with_speed(0.0, 0.0)
            .with_direction(Vec3::Z, 179.0)
            .with_motion(MotionKind::Flock {
                n: 8,
                sep: 0.2,
                align: 0.1,
                coh: 0.9,
                seek_light: 0.0,
            });
        let em = Emitter::from_spec(&spec);
        let mut pool = ParticlePool::with_capacity(16);
        let mut rng = ParticleRng::seed(21);
        tick_emitters(std::slice::from_ref(&em), &mut pool, 1.0 / 30.0, &mut rng);
        let spread_of = |pool: &ParticlePool| -> f32 {
            let pts: Vec<Vec3> = (0..pool.capacity())
                .filter(|&i| pool.alive[i])
                .map(|i| pool.pos[i])
                .collect();
            let centroid = pts.iter().fold(Vec3::ZERO, |a, &p| a + p) / pts.len() as f32;
            pts.iter().map(|p| (*p - centroid).length()).sum::<f32>() / pts.len() as f32
        };
        let spread0 = spread_of(&pool);
        for _ in 0..90 {
            tick_emitters(std::slice::from_ref(&em), &mut pool, 1.0 / 30.0, &mut rng);
        }
        let spread1 = spread_of(&pool);
        assert!(
            spread1 <= spread0 + 1e-3,
            "cohesion-dominated flocking must not let the group spread out (got {spread0} -> {spread1})"
        );
    }

    #[test]
    fn floor_bounce_reflects_and_damps_vertical_velocity() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(0.0, 0.0, 1.0)))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(5.0, 0.0)
            .with_speed(3.0, 0.0)
            .with_direction(Vec3::new(0.0, 0.0, -1.0), 0.0)
            .with_accel(Vec3::new(0.0, 0.0, -9.0))
            .with_bounce(0.5);
        let em = Emitter::from_spec(&spec);
        assert_eq!(
            em.floor_z, 1.0,
            "from_spec bakes floor_z from the world origin"
        );
        let mut pool = ParticlePool::with_capacity(4);
        let mut rng = ParticleRng::seed(4);
        tick_emitters(std::slice::from_ref(&em), &mut pool, 1.0 / 30.0, &mut rng);
        let i = (0..pool.capacity())
            .find(|&i| pool.alive[i])
            .expect("one live particle");
        // Force the particle down through the floor and confirm the NEXT tick
        // reflects it back up with damped speed rather than tunnelling through.
        pool.pos[i].z = em.floor_z - 0.1;
        pool.vel[i] = Vec3::new(0.0, 0.0, -4.0);
        let floor_z = em.floor_z;
        tick_emitters(&[em], &mut pool, 1.0 / 30.0, &mut rng);
        assert!(
            pool.pos[i].z >= floor_z,
            "bounce must not leave the particle under the floor"
        );
        assert!(
            pool.vel[i].z > 0.0,
            "a bounce must reverse the vertical velocity"
        );
        assert!(
            pool.vel[i].z < 4.0,
            "restitution < 1.0 must damp the bounce speed"
        );
    }

    #[test]
    fn lifetime_speed_size_distributions_are_deterministic_and_bounded() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::ZERO))
            .with_rate(30.0)
            .with_lifetime(1.0, 0.0)
            .with_distributions(
                Some(Distribution::Uniform { lo: 0.5, hi: 1.5 }),
                Some(Distribution::Uniform { lo: 1.0, hi: 2.0 }),
                Some(Distribution::Uniform { lo: 0.2, hi: 0.8 }),
            );
        let em = Emitter::from_spec(&spec);
        let run = || run_ticks(std::slice::from_ref(&em), 128, 77, 1.0 / 30.0, 10);
        let pool_a = run();
        let pool_b = run();
        for i in 0..pool_a.capacity() {
            assert_eq!(pool_a.alive[i], pool_b.alive[i]);
            if pool_a.alive[i] {
                assert_eq!(
                    pool_a.life[i], pool_b.life[i],
                    "lifetime_dist must be deterministic"
                );
                assert!(
                    (0.5..=1.5).contains(&pool_a.life[i]),
                    "lifetime_dist must stay within its authored range, got {}",
                    pool_a.life[i]
                );
            }
        }
    }

    #[test]
    fn color_jitter_shifts_the_ramp_but_stays_deterministic() {
        let base = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::ZERO))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(1.0, 0.0)
            .with_colors(0xC83232, 0xC83232, 0xC83232);
        let jittered = base.clone().with_color_jitter(ColorJitter {
            h_deg: 60.0,
            s: 0.5,
            l: 0.0,
            dance: 0.0,
        });
        let em_plain = Emitter::from_spec(&base);
        let em_jit = Emitter::from_spec(&jittered);

        let sample = |em: &Emitter| {
            let pool = run_ticks(std::slice::from_ref(em), 4, 13, 1.0 / 30.0, 1);
            let i = (0..pool.capacity()).find(|&i| pool.alive[i]).unwrap();
            pool.ramp[i].sample(0.5)
        };
        let plain_color = sample(&em_plain);
        let jit_a = sample(&em_jit);
        let jit_b = sample(&em_jit);
        assert_eq!(
            jit_a, jit_b,
            "color_jitter must be deterministic for the same seed"
        );
        assert_ne!(
            jit_a, plain_color,
            "color_jitter must actually shift the ramp"
        );
    }

    #[test]
    fn glyph_pick_random_holds_one_member_for_the_particles_life() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::ZERO))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(1.0, 0.0)
            .with_glyph_pick(GlyphPick::Random);
        let em = Emitter::from_spec(&spec);
        let pool = run_ticks(&[em], 4, 2, 1.0 / 30.0, 1);
        let i = (0..pool.capacity()).find(|&i| pool.alive[i]).unwrap();
        assert_ne!(
            pool.glyph_fixed[i], NO_FIXED_GLYPH,
            "Random pick must fix a glyph at birth"
        );
    }

    #[test]
    fn glyph_source_text_cycles_characters_across_births() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::ZERO))
            .with_rate(3.0)
            .with_lifetime(10.0, 0.0)
            .with_glyph_source(GlyphSource::Text("AB".to_string()));
        let em = Emitter::from_spec(&spec);
        let pool = run_ticks(&[em], 16, 1, 1.0, 1);
        let glyphs: std::collections::BTreeSet<u32> = (0..pool.capacity())
            .filter(|&i| pool.alive[i])
            .map(|i| pool.literal_glyph[i])
            .collect();
        assert!(
            glyphs.contains(&(b'A' as u32)),
            "word particles must cycle through 'A'"
        );
        assert!(
            glyphs.contains(&(b'B' as u32)),
            "word particles must cycle through 'B'"
        );
    }

    #[test]
    fn dust_visibility_lit_is_gated_by_nearby_lights() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(0.0, 0.0, 1.0)))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(1.0, 0.0)
            .with_visibility(DustVisibility::Lit { gain: 3.0 });

        // No lights baked in (`from_spec`) → invisible (vis == 0).
        let em_dark = Emitter::from_spec(&spec);
        let pool_dark = run_ticks(&[em_dark], 4, 6, 1.0 / 30.0, 1);
        let i = (0..pool_dark.capacity())
            .find(|&i| pool_dark.alive[i])
            .unwrap();
        assert_eq!(
            pool_dark.vis[i], 0.0,
            "Lit visibility with no lights must be fully dark"
        );

        // A bright light right at the origin (`from_spec_lit`) → visible.
        let light = PointLight {
            pos: Vec3::new(0.0, 0.0, 1.0),
            color: 0xFFFFFF,
            radius: 4.0,
            intensity: 2.0,
        };
        let em_lit = Emitter::from_spec_lit(&spec, &[light]);
        let pool_lit = run_ticks(&[em_lit], 4, 6, 1.0 / 30.0, 1);
        let j = (0..pool_lit.capacity())
            .find(|&i| pool_lit.alive[i])
            .unwrap();
        assert!(
            pool_lit.vis[j] > 0.0,
            "Lit visibility near a light must be visible"
        );
    }

    #[test]
    fn trail_buffer_fills_as_a_particle_moves() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::ZERO))
            .with_rate(1.0)
            .with_burst(1)
            .with_lifetime(5.0, 0.0)
            .with_speed(2.0, 0.0)
            .with_direction(Vec3::new(1.0, 0.0, 0.0), 0.0)
            .with_trail(TrailSpec {
                count: 4,
                fade: 0.6,
                dir_glyphs: true,
            });
        let em = Emitter::from_spec(&spec);
        let pool = run_ticks(&[em], 4, 8, 1.0 / 30.0, 5);
        let i = (0..pool.capacity())
            .find(|&i| pool.alive[i])
            .expect("one live trailing particle");
        assert!(
            pool.trail_n[i] >= 2,
            "a moving trailed particle must accumulate trail history"
        );
        // The trail's ring is FIFO-shifted, so the current head (`buf[0]`) must be
        // strictly ahead (further +x) than the next-oldest sample.
        let buf = pool.trail_buf[i];
        assert!(
            buf[0].x >= buf[1].x,
            "trail head must be at or ahead of the next-oldest sample"
        );
    }

    #[test]
    fn b6_overlapping_particles_accumulate_into_a_density_glyph() {
        // Two hand-placed STANDARD particles landing in the SAME screen cell must
        // no longer last-write-wins: the resolved glyph must be the density-ramp
        // glyph for `n=2`, not either particle's own glyph.
        let cols = 20usize;
        let rows = 12usize;
        let fov = Fov::default();
        let cam = CameraPose {
            pos: Vec3::new(0.0, 0.0, 0.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut pool = ParticlePool::with_capacity(4);
        for i in 0..2 {
            pool.pos[i] = Vec3::new(0.0, 5.0, 0.5);
            pool.vel[i] = Vec3::ZERO;
            pool.age[i] = 0.0;
            pool.life[i] = 1.0;
            pool.ramp[i] = ColorRamp::two(0xFF0000, 0xFF0000);
            pool.family[i] = GROWTH_DOT;
            pool.alive[i] = true;
        }
        let opaque = Layer::new(cols, rows, re_core::LayerBlend::Opaque);
        let mut out = Layer::new(cols, rows, re_core::LayerBlend::GlyphOnly);
        splat_particles(&pool, &cam, &fov, &opaque, &mut out);
        let cell = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .find(|&(c, r)| out.is_present(c, r))
            .expect("both particles must land in some cell");
        let got = out.get(cell.0, cell.1).unwrap();
        assert_eq!(
            got.glyph,
            density_ramp_glyph(2),
            "two overlapping particles must resolve to the n=2 density glyph, not one's own"
        );
    }
}
