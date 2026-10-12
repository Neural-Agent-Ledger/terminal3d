//! `re_world::emitter_spec` — the pure-data, engine-agnostic `EmitterSpec`.
//!
//! This is the CONTRACT type: `re_level` authors it (parses `.lvl` keys into it),
//! `re_render` resolves it into a live `Emitter` via `Emitter::from_spec`, and
//! `re_game` wires the two together. `re_world` is the only crate that all three
//! already depend on, so the type lives here — pure data, no render deps.
//!
//! # `.lvl` key mapping
//!
//! Each field's doc comment names the `.lvl` key `re_level` will parse. Section
//! header is `[emitter <id>]` (one section per emitter in the level). Example:
//!
//! ```text
//! [emitter torch_1]
//! origin_world  = 3.5 2.0 1.2
//! colors        = 0xFFD27F 0xFF7A1A 0x6E1200
//! color_mode    = RampOverAge
//! family        = growth_dot
//! bold          = ByAge
//! direction     = 0.0 0.0 1.0
//! spread_deg    = 30.0
//! size          = 1.0
//! growth        = 1.5
//! rate          = 24.0
//! speed         = 0.9
//! speed_spread  = 0.2
//! lifetime      = 2.2
//! lifetime_spread = 0.3
//! accel         = 0.0 0.0 0.35
//! max_particles = 256
//! seed          = 42
//! blend         = KeepBg
//! ```
//!
//! # No render types
//!
//! `re_world` does NOT depend on `re_render`. `GlyphFamily` lives in `re_render`
//! and is referenced here as a `String` name (e.g. `"growth_dot"`). `re_render`
//! resolves it via `glyph_family::family_by_name`. An explicit `glyph_set` escape
//! hatch lets a level bypass the family registry entirely.

use glam::Vec3;

// ---------------------------------------------------------------------------
// Emitter-variety extensions
// ---------------------------------------------------------------------------
//
// Every type/field below is ADDITIVE and opt-in: its `Default` reproduces
// TODAY'S behaviour exactly (Ballistic motion, no distributions, no jitter, no
// trail, no bounce, always-visible, family-driven glyphs), so a level/spec that
// never sets one of these keys renders byte-identical to before this extension.
// `EmitterSpec` is a plain-data, non-`#[non_exhaustive]` struct constructed via
// struct-literal in a couple of call sites outside this crate (e.g.
// `re_level`'s `[effects] snow` synthesis) — those sites use
// `..Default::default()` for exactly this reason, so a field ADDED here can
// never break their compile.

/// A scalar sampling law used by [`EmitterSpec::lifetime_dist`] /
/// [`speed_dist`](EmitterSpec::speed_dist) / [`size_dist`](EmitterSpec::size_dist)
/// (spec §4.3 "Distributions"). `None` on the owning field means "use the plain
/// mean±spread pair instead" (today's behaviour) — a `Distribution` is an
/// override, not a replacement of those fields.
///
/// Every sample is a PURE function of a per-particle uniform draw `u ∈ [0,1)`
/// (itself `hash64(particle_seed, PARAM_ID)`-derived by the caller), so two runs
/// with the same particle seed produce the same sample — no fresh RNG state.
///
/// `.lvl` value grammar (accepted wherever a distribution-capable key is parsed):
/// `a..b` (uniform), `m~s` (Gaussian, clamped at ±3σ), `a..b^k` (power law).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distribution {
    /// Uniform in `[lo, hi]`.
    Uniform {
        /// Lower bound (inclusive).
        lo: f32,
        /// Upper bound (inclusive).
        hi: f32,
    },
    /// Gaussian with `mean`/`std`, clamped to `mean ± 3·std` (never an unbounded
    /// tail reaching a degenerate particle).
    Gaussian {
        /// Distribution mean.
        mean: f32,
        /// Standard deviation.
        std: f32,
    },
    /// Power-law in `[lo, hi]`: `lo + (hi-lo)·u^k`. `k > 1` biases toward `lo`
    /// (many faint/short, few bright/long — the starfield-magnitude shape);
    /// `k < 1` biases toward `hi`.
    PowerLaw {
        /// Lower bound.
        lo: f32,
        /// Upper bound.
        hi: f32,
        /// Power exponent; `k > 1` biases toward `lo`, `k < 1` toward `hi`.
        k: f32,
    },
}

impl Distribution {
    /// Sample at uniform draws `u0, u1 ∈ [0,1)` (two independent draws so
    /// [`Gaussian`](Self::Gaussian) can use Box–Muller; the other variants only
    /// consume `u0`). Pure — same `(u0, u1)` always returns the same value.
    pub fn sample(&self, u0: f32, u1: f32) -> f32 {
        match *self {
            Distribution::Uniform { lo, hi } => lo + (hi - lo) * u0.clamp(0.0, 1.0),
            Distribution::Gaussian { mean, std } => {
                // Box–Muller, clamped at 3σ (spec: "clamped at 3σ").
                let u0 = u0.clamp(1e-6, 1.0 - 1e-6);
                let u1 = u1.clamp(0.0, 1.0);
                let r = (-2.0 * u0.ln()).max(0.0).sqrt();
                let z = r * (std::f32::consts::TAU * u1).cos();
                mean + std * z.clamp(-3.0, 3.0)
            }
            Distribution::PowerLaw { lo, hi, k } => {
                let u0 = u0.clamp(0.0, 1.0);
                lo + (hi - lo) * u0.powf(k.max(0.001))
            }
        }
    }
}

/// Per-particle colour perturbation applied once at birth (spec §4.3
/// "Colour jitter, Brogue-style"): a fixed `±` range per HSL channel plus a
/// `dance` knob reserved for a future time-varying re-roll (today `dance` is
/// accepted and stored but not yet animated — see [`EmitterSpec::color_jitter`]
/// docs). Implemented in HSL (not Oklch) because the runtime Oklab helpers do
/// not exist yet; this is a documented, self-contained substitute, not a
/// silent shortcut — see `docs/EMITTERS.md`.
///
/// `.lvl` key: `color_jitter = h±<deg> s±<0..1> l±<0..1> dance=<0..1>`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorJitter {
    /// Symmetric hue perturbation, degrees.
    pub h_deg: f32,
    /// Symmetric saturation perturbation, `[0,1]` fraction.
    pub s: f32,
    /// Symmetric lightness perturbation, `[0,1]` fraction.
    pub l: f32,
    /// Reserved: fraction of the jitter range re-rolled over time. Accepted and
    /// clamped to `[0,1]`; not yet animated (static per-particle jitter today).
    pub dance: f32,
}

impl Default for ColorJitter {
    fn default() -> Self {
        ColorJitter {
            h_deg: 0.0,
            s: 0.0,
            l: 0.0,
            dance: 0.0,
        }
    }
}

/// Whether a particle's glyph is chosen by its age (today's behaviour) or once
/// at birth and held for life (spec §4.5 `glyph_pick = random`).
///
/// `.lvl` key: `glyph_pick = Sequential | Random`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GlyphPick {
    /// Age-driven (today's `family.pick(glyph_drive(...))`).
    #[default]
    Sequential,
    /// One random member of the family, chosen at birth from the particle's
    /// deterministic seed, held for the particle's whole life.
    Random,
}

/// Where a particle's glyph sequence comes from (spec §4.2 W12 "word
/// particles"). `Family` (default) is today's family/`glyph_set` path
/// unchanged. `Text` breaks a string apart: each spawned particle claims the
/// NEXT character (`birth_index % chars.len()`, deterministic, cycling if the
/// emitter outlives one full pass), so a steady stream of particles reads as
/// the word dissolving into its own letters over time.
///
/// `.lvl` key: `glyph_source = family | text "<string>"`
#[derive(Clone, Debug, PartialEq, Default)]
pub enum GlyphSource {
    /// Use [`EmitterSpec::family`] / [`glyph_set`](EmitterSpec::glyph_set)
    /// (today's behaviour).
    #[default]
    Family,
    /// Each particle's glyph is one character of `0`, chosen by
    /// `birth_index % chars().count()`. Reliable-glyph filtering is the
    /// author's responsibility (unreliable glyphs are not rejected here, same
    /// as `glyph_set`).
    Text(String),
}

/// A per-particle motion trail: the last few positions, drawn as a fading tail
/// behind the head with an optional direction glyph (spec §4.2 W4).
///
/// `.lvl` key: `trail = <count> fade=<0..1> dir_glyphs=<on|off>`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailSpec {
    /// How many trailing positions to remember (clamped to `[1, 6]` — see
    /// `re_render::emitter::TRAIL_CAP`).
    pub count: u8,
    /// Per-step brightness falloff along the tail, `[0,1]` (0 = tail invisible
    /// past the head, 1 = no falloff).
    pub fade: f32,
    /// When true, the HEAD glyph is chosen from the projected screen-space
    /// velocity direction (`─ │ ╱ ╲`) instead of the family/age glyph.
    pub dir_glyphs: bool,
}

impl Default for TrailSpec {
    fn default() -> Self {
        TrailSpec {
            count: 4,
            fade: 0.6,
            dir_glyphs: true,
        }
    }
}

/// Whether a particle always renders (today's behaviour) or only where lit
/// (spec §4.2 W9 "dust in light shafts"). `Lit` particles are gated by a light
/// sample taken ONCE at birth from whatever lights were baked into the runtime
/// `Emitter` (see `re_render::Emitter::from_spec_lit`) — a documented
/// simplification (dust drifts slowly, so a birth-time sample reads correctly
/// for its lifetime) rather than a continuous per-frame re-sample.
///
/// `.lvl` key: `visibility = always | lit gain=<f32>`
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum DustVisibility {
    /// Always renders — today's behaviour.
    #[default]
    Always,
    /// Renders only where the birth-time light sample, scaled by `gain`, is
    /// above a small floor; fully invisible in the dark.
    Lit {
        /// Scales the birth-time light sample before the visibility floor.
        gain: f32,
    },
}

/// The particle's bulk MOTION MODEL (spec §4.2). `Ballistic` (default) is
/// today's vel/accel/turbulence/swirl integration, byte-for-byte unchanged.
/// Every other variant REPLACES the per-tick velocity/position update for
/// particles spawned by this emitter — see `re_render::emitter::tick_emitters`.
///
/// `.lvl` key: `motion = ballistic | curl scale=… strength=… evolve=… | attract
/// X Y Z strength=… swirl=… | lorenz scale=… | flock n=… sep=… align=… coh=…
/// seek_light=… | ripple speed=…`
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum MotionKind {
    /// Today's vel/accel ballistic integration + turbulence/swirl. Default.
    #[default]
    Ballistic,
    /// Divergence-free 2D curl-noise drift in the horizontal plane (spec §4.2
    /// W1), layered ON TOP of the ballistic accel (so buoyancy/gravity still
    /// applies) — a documented simplification of Bridson's 3D curl noise,
    /// restricted to the horizontal plane (fits smoke/spores/dust drifting near
    /// level, and needs a third of the noise evaluations).
    Curl {
        /// Noise-field spatial scale.
        scale: f32,
        /// Drift strength.
        strength: f32,
        /// Rate the noise field evolves over time.
        evolve: f32,
    },
    /// Pulled toward `target` by an inverse-square-ish force, plus a tangential
    /// `swirl` term for an orbit rather than a straight fall-in (spec §4.2 W2).
    Attract {
        /// World-space attraction point.
        target: Vec3,
        /// Pull strength.
        strength: f32,
        /// Tangential (orbiting) component.
        swirl: f32,
    },
    /// The classic Lorenz chaotic attractor (σ=10, ρ=28, β=8/3, RK2-integrated),
    /// scaled into world cells (spec §4.2 W2 "optional lorenz"). The particle's
    /// position IS the (scaled) attractor state; `vel`/`accel` are ignored.
    Lorenz {
        /// Scales the attractor state into world cells.
        scale: f32,
    },
    /// Boids (Reynolds separation/alignment/cohesion) among this emitter's own
    /// live particles, capped at `n`, with an optional pull toward the nearest
    /// baked light (`seek_light`, needs `re_render::Emitter::from_spec_lit` —
    /// `0.0` disables it) (spec §4.2 W3).
    Flock {
        /// Max flockmates considered per particle.
        n: u32,
        /// Separation weight.
        sep: f32,
        /// Alignment weight.
        align: f32,
        /// Cohesion weight.
        coh: f32,
        /// Pull toward the nearest baked light; `0.0` disables it.
        seek_light: f32,
    },
    /// An expanding ring on the horizontal plane centred on the emitter's
    /// origin: `radius(age) = speed·age`, angle fixed per particle from its
    /// deterministic seed (spec §4.2 W10). `vel`/`accel` are ignored; height
    /// stays at the origin's `z`. See `re_render::ripple`.
    Ripple {
        /// Ring expansion rate, world cells per second.
        speed: f32,
    },
}

// ---------------------------------------------------------------------------
// EmitterOrigin
// ---------------------------------------------------------------------------

/// Where a particle emitter births particles.
///
/// `.lvl` keys: `origin_world = X Y Z`  or  `origin_screen = COL ROW`
/// (mutually exclusive; `origin_world` takes precedence if both are set).
#[derive(Clone, Debug, PartialEq)]
pub enum EmitterOrigin {
    /// A world-space point in cell units. Particles project through the camera
    /// and are depth-tested against solid geometry (occluded by walls).
    World(Vec3),
    /// A fixed screen cell `(col, row)` in fractional terminal coordinates.
    /// Particles are pinned to the viewport (HUD sparks / screen-space overlays)
    /// and are never occluded by geometry.
    Screen {
        /// Fractional terminal column (0 = left edge).
        col: f32,
        /// Fractional terminal row (0 = top edge).
        row: f32,
    },
}

impl Default for EmitterOrigin {
    /// Defaults to the world origin — a sensible centre-room anchor when
    /// the `.lvl` author has not specified a position yet.
    fn default() -> Self {
        EmitterOrigin::World(Vec3::new(3.5, 3.5, 1.5))
    }
}

// ---------------------------------------------------------------------------
// ColorMode
// ---------------------------------------------------------------------------

/// How the three authored colour stops (`.lvl` key: `colors = 0xRRGGBB 0xRRGGBB 0xRRGGBB`)
/// are applied to the particle's foreground colour over its lifetime.
///
/// `.lvl` key: `color_mode = RampOverAge | RandomPick | Fixed`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// Linearly interpolate the three stops over the particle's normalised age
    /// `t ∈ \[0, 1\]`: `stop[0]` at birth, `stop[1]` at mid-life, `stop[2]` at death.
    /// The most expressive and the default. `re_render` maps this to a
    /// `ColorRamp::three(colors[0], colors[1], colors[2])`.
    #[default]
    RampOverAge,
    /// Pick one of the three stops at random at birth and keep it for the
    /// particle's lifetime (colour variety within a spray).
    RandomPick,
    /// Every particle uses `colors[0]` for its entire lifetime (ignores stops
    /// 1 and 2). Cheapest; good for a single-tint mist.
    Fixed,
}

// ---------------------------------------------------------------------------
// BoldMode
// ---------------------------------------------------------------------------

/// Whether particles render with bold/bright terminal attribute, and what drives it.
///
/// `.lvl` key: `bold = Off | On | ByProximity | ByAge`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BoldMode {
    /// Never bold — standard weight throughout lifetime. Default.
    #[default]
    Off,
    /// Always bold — the particle glows bright for its entire lifetime.
    On,
    /// Bold while the particle is close to the camera (small depth); fades to
    /// normal weight beyond a renderer-defined proximity threshold.
    ByProximity,
    /// Bold while young (birth → ~20% of lifetime), then normal — a quick
    /// flash at spawn, settling to a softer look.
    ByAge,
}

// ---------------------------------------------------------------------------
// EmitterBlend
// ---------------------------------------------------------------------------

/// How the particle splat composites over the layers beneath it.
///
/// Mirrors the intent of `re_core::LayerBlend` without depending on `re_render`.
/// `re_render::Emitter::from_spec` maps this to `re_core::LayerBlend`.
///
/// `.lvl` key: `blend = KeepBg | Opaque`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EmitterBlend {
    /// Write only the glyph and foreground colour; the background is taken from
    /// the layer beneath (hologram / wall bg shows through). Default — the
    /// `GlyphOnly` blend of the composite spec §2.
    #[default]
    KeepBg,
    /// The particle cell is fully opaque: glyph + fg + bg all replace the
    /// layer beneath. Useful for bright, solid sparks over a dark bg.
    Opaque,
}

// ---------------------------------------------------------------------------
// EmitterKind
// ---------------------------------------------------------------------------

/// The high-level rendering behaviour of an emitter — what kind of effect it is.
///
/// This selects the compositing/visual model the renderer uses for the emitter,
/// orthogonal to all the per-particle tuning fields on [`EmitterSpec`].
///
/// `.lvl` key: `kind = Standard | GodRays`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EmitterKind {
    /// A normal particle emitter: glyph splats with colour, depth-tested and
    /// composited per [`blend`](EmitterSpec::blend). The default — every existing
    /// emitter is `Standard`.
    #[default]
    Standard,
    /// Colour-only additive light shafts ("god rays"): the emitter contributes a
    /// glow that is ADDED to the colours already in the scene, drawing no glyphs of
    /// its own. Rendered via `re_core::LayerBlend::ColorAdd`, which adds `fg`/`bg`
    /// (clamped to 255 per channel) while keeping the destination glyph and style —
    /// so the shaft tints the scene without erasing detail.
    GodRays,
}

// ---------------------------------------------------------------------------
// EmitterSpec
// ---------------------------------------------------------------------------

/// The pure-data, authored description of a particle emitter.
///
/// A CONTRACT type: authored in `.lvl` files (parsed by `re_level`), carried by
/// `re_world` (as plain data on the `Map` or a parallel list), and resolved into a
/// live `re_render::Emitter` by `Emitter::from_spec`. `re_game` bridges the two.
///
/// All fields have documented `.lvl` key names so `re_level`'s parser knows exactly
/// which key maps to which field. Unspecified keys fall through to the `Default`
/// (a gentle upward haze — immediately visible and clearly working).
///
/// # Coordinate conventions
///
/// * `direction` is in world space: `(0, 0, 1)` = straight up, `(0, 1, 0)` = south.
/// * `speed` and `accel` are in cell-units per second.
/// * Colours are packed `0x00RRGGBB` (the same convention as `MaterialDef::fg`).
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterSpec {
    // ── KIND ────────────────────────────────────────────────────────────────
    /// The high-level rendering behaviour of this emitter.
    ///
    /// `Standard` (the default) is an ordinary glyph-splat particle emitter;
    /// `GodRays` is a colour-only additive light shaft (no glyphs) rendered via
    /// `re_core::LayerBlend::ColorAdd`. See [`EmitterKind`].
    ///
    /// `.lvl` key: `kind = Standard | GodRays`
    pub kind: EmitterKind,

    // ── ORIGIN ──────────────────────────────────────────────────────────────
    /// Where particles are born.
    ///
    /// `.lvl` keys: `origin_world = X Y Z`  or  `origin_screen = COL ROW`
    pub origin: EmitterOrigin,

    // ── COLOUR ──────────────────────────────────────────────────────────────
    /// Three colour stops, packed `0x00RRGGBB`. Interpretation depends on
    /// [`color_mode`](Self::color_mode).
    ///
    /// `.lvl` key: `colors = 0xRRGGBB 0xRRGGBB 0xRRGGBB`
    pub colors: [u32; 3],

    /// How the three colour stops are applied over the particle's lifetime.
    ///
    /// `.lvl` key: `color_mode = RampOverAge | RandomPick | Fixed`
    pub color_mode: ColorMode,

    // ── GLYPHS ──────────────────────────────────────────────────────────────
    /// Glyph-family name resolved by `re_render::glyph_family::family_by_name`.
    ///
    /// The canonical family names from the `re_render` library:
    /// `haze_o`, `haze_dot`, `haze_wave`, `haze_colon`, `haze_bar`,
    /// `growth_dot`, `growth_density`, `growth_bars`,
    /// `anim_spin`, `anim_arc`, `anim_sweep`,
    /// `lean_fill`, `lean_stub`.
    ///
    /// An unknown name falls back to `"growth_dot"` at resolve time.
    ///
    /// `.lvl` key: `family = <name>`
    pub family: String,

    /// Optional explicit glyph sequence override. When `Some`, bypasses
    /// `family` entirely and uses these characters in order (indexed by
    /// normalised age, same as a Growth family). Useful for a bespoke
    /// one-off sequence not in the library.
    ///
    /// `.lvl` key: `glyph_set = <char> <char> …`  (space-separated Unicode scalars)
    pub glyph_set: Option<Vec<char>>,

    // ── BOLD ────────────────────────────────────────────────────────────────
    /// Whether / when to render the particle's terminal cell as bold/bright.
    ///
    /// `.lvl` key: `bold = Off | On | ByProximity | ByAge`
    pub bold: BoldMode,

    // ── DIRECTION & SPREAD ──────────────────────────────────────────────────
    /// Mean emission direction (world-space unit vector). Does not need to be
    /// pre-normalised — `re_render` normalises it at resolve time. Zero vector
    /// ⇒ omni (treated the same as `spread_deg = 180`).
    ///
    /// `.lvl` key: `direction = X Y Z`
    pub direction: Vec3,

    /// Cone half-angle in degrees `[0, 180]`. `0` = pencil-thin beam along
    /// `direction`; `180` = omnidirectional. Clamped to `[0, 180]` by
    /// [`EmitterSpec::validate`].
    ///
    /// `.lvl` key: `spread_deg = <degrees>`
    pub spread_deg: f32,

    // ── SIZE & GROWTH ────────────────────────────────────────────────────────
    /// Base size scalar at birth. Drives the Growth-family glyph index at `t=0`.
    /// `1.0` = default (the smallest growth member); larger values start bigger.
    ///
    /// `.lvl` key: `size = <f32>`
    pub size: f32,

    /// Size scale over age: the effective size at `t=1` is `size * growth`.
    /// `1.0` = no change; `>1.0` = particle grows; `<1.0` = shrinks.
    /// Growth-family glyph index is driven by `lerp(size, size * growth, t)`,
    /// normalised into the family's range.
    ///
    /// `.lvl` key: `growth = <f32>`
    pub growth: f32,

    // ── FREQUENCY ───────────────────────────────────────────────────────────
    /// Particles spawned per second (steady state, fractional-accumulated).
    ///
    /// `.lvl` key: `rate = <f32>`
    pub rate: f32,

    /// Optional one-shot burst: if `Some(n)`, emit exactly `n` particles at
    /// emitter activation and then set `rate = 0`. `None` = continuous stream.
    ///
    /// `.lvl` key: `burst = <u32>`
    pub burst: Option<u32>,

    // ── SPEED ────────────────────────────────────────────────────────────────
    /// Mean initial speed along `direction` (cell-units per second).
    ///
    /// `.lvl` key: `speed = <f32>`
    pub speed: f32,

    /// Symmetric speed spread: actual speed = `speed ± speed_spread` (uniform).
    ///
    /// `.lvl` key: `speed_spread = <f32>`
    pub speed_spread: f32,

    // ── PATH RANDOMISATION ───────────────────────────────────────────────────
    /// Per-step turbulence amplitude (cell-units per second of random impulse).
    /// `0.0` = perfectly smooth paths; positive values add noise each integration
    /// step — embers drift, haze wanders.
    ///
    /// `.lvl` key: `path_jitter = <f32>`
    pub path_jitter: f32,

    /// Deterministic RNG seed. Same seed + same emitter state → identical
    /// particle trajectories every run (golden-frame safe).
    ///
    /// `.lvl` key: `seed = <u64>`
    pub seed: u64,

    /// Curl/rotational drift strength (cell-units per second). Positive values
    /// impart a counter-clockwise swirl (viewed from above); negative = clockwise.
    /// `0.0` = no swirl (default).
    ///
    /// `.lvl` key: `swirl = <f32>`
    pub swirl: f32,

    // ── LIFETIME ────────────────────────────────────────────────────────────
    /// Mean particle lifetime in seconds. Must be `> 0`.
    ///
    /// `.lvl` key: `lifetime = <f32>`
    pub lifetime: f32,

    /// Symmetric lifetime spread: actual lifetime = `lifetime ± lifetime_spread`
    /// (uniform, clamped so the result is always `≥ 0.01 s`).
    ///
    /// `.lvl` key: `lifetime_spread = <f32>`
    pub lifetime_spread: f32,

    // ── FORCES ───────────────────────────────────────────────────────────────
    /// Constant acceleration applied every integration step (cell-units / sec²).
    /// `(0, 0, -9.8)` simulates gravity; `(0, 0, +0.4)` simulates buoyancy.
    ///
    /// `.lvl` key: `accel = X Y Z`
    pub accel: Vec3,

    // ── CAPACITY ─────────────────────────────────────────────────────────────
    /// Hard per-emitter particle cap. Spawning is silently skipped when the live
    /// count hits this ceiling (never a realloc). Must be `≥ 1`.
    ///
    /// `.lvl` key: `max_particles = <u32>`
    pub max_particles: u32,

    // ── BLEND ────────────────────────────────────────────────────────────────
    /// How this emitter's splat composites over the layers beneath it.
    ///
    /// `.lvl` key: `blend = KeepBg | Opaque`
    pub blend: EmitterBlend,

    // ── PROVENANCE ──────────────────────────────────────────────────────────
    /// `true` when this spec was SYNTHESIZED at load time from a non-emitter
    /// `.lvl` knob (today: `[effects] snow`, expanded by `re_level` into a
    /// lattice of ambient emitters) rather than authored via a real `[emitter]`
    /// section.
    ///
    /// This is a pure in-memory provenance marker — there is no `.lvl` key for
    /// it, and no parser ever sets it `true` while reading a `[emitter]`
    /// section (so every authored/edited spec, including one an editor pushes
    /// onto `Level::emitters`, defaults to `false` regardless of WHERE in the
    /// list it lands). It exists so a writer (`re_voximport::to_lvl_text`) and
    /// an editor (`ts_editor`) can filter synthesized emitters out of what they
    /// persist/display — re-emitting them would double the source effect on
    /// the next load, and an editor cannot usefully duplicate/delete/inspect an
    /// object that regenerates itself from a knob. The renderer does NOT
    /// filter on this: a synthesized emitter renders exactly like any other.
    pub synthesized: bool,

    // ── VARIETY & RANDOMISATION ─────────────────────────────────────────────
    // Every field below defaults to "today's behaviour, unchanged" — see each
    // type's docs. All are additive; `EmitterSpec::default()` sets every one to
    // its neutral value.
    /// The bulk motion model. `.lvl` key: `motion`.
    pub motion: MotionKind,

    /// Overrides `lifetime`/`lifetime_spread` when `Some` (spec §4.3
    /// distributions). `.lvl`: the distribution grammar on the `lifetime` key.
    pub lifetime_dist: Option<Distribution>,
    /// Overrides `speed`/`speed_spread` when `Some`. `.lvl`: on `speed`.
    pub speed_dist: Option<Distribution>,
    /// Overrides `size` (birth size) when `Some`. `.lvl`: on `size`.
    pub size_dist: Option<Distribution>,

    /// Scales every "spread" quantity (`lifetime_spread`, `speed_spread`,
    /// `path_jitter`, `color_jitter`'s ranges) by this factor. `1.0` = author's
    /// values unchanged (default). `.lvl` key: `randomize`.
    pub randomize: f32,

    /// Per-particle colour perturbation, re-rolled once at birth. `None` = no
    /// jitter (default). `.lvl` key: `color_jitter`.
    pub color_jitter: Option<ColorJitter>,

    /// An explicit N-stop (2..=8) colour ramp over age, overriding `colors`/
    /// `color_mode` when `Some`. `.lvl` key: `color_ramp = 0xRRGGBB …` (2-8
    /// stops, evenly spaced over `[0,1]`).
    pub color_ramp: Option<Vec<u32>>,

    /// Whether the glyph is age-driven (today) or a fixed per-particle random
    /// pick (spec §4.5). `.lvl` key: `glyph_pick`.
    pub glyph_pick: GlyphPick,

    /// Where the glyph sequence comes from — the family/`glyph_set` path
    /// (default) or an authored word broken apart letter by letter (spec §4.2
    /// W12). `.lvl` key: `glyph_source`.
    pub glyph_source: GlyphSource,

    /// A fading motion trail behind each particle. `None` = no trail (today's
    /// behaviour, default). `.lvl` key: `trail`.
    pub trail: Option<TrailSpec>,

    /// Floor-bounce restitution coefficient (spec §4.2 W5 sparks). `None` = no
    /// bounce (default, today's behaviour — a particle simply falls/culls).
    /// The bounce floor is the emitter's own origin height (`origin.z` for a
    /// `World` anchor; disabled for `Screen`). `.lvl` key: `bounce`.
    pub bounce: Option<f32>,

    /// Whether the particle always renders or only where lit (spec §4.2 W9).
    /// `.lvl` key: `visibility`.
    pub visibility: DustVisibility,
}

impl Default for EmitterSpec {
    /// A gentle upward haze anchored near the centre of the test room — the
    /// "nothing authored yet but visibly working" default. All fields are sane
    /// and pass [`validate`](Self::validate).
    fn default() -> Self {
        EmitterSpec {
            kind: EmitterKind::Standard,
            origin: EmitterOrigin::default(),
            colors: [0x9FB8D8, 0x4A7AA8, 0x2A3A52],
            color_mode: ColorMode::RampOverAge,
            family: String::from("haze_dot"),
            glyph_set: None,
            bold: BoldMode::Off,
            direction: Vec3::new(0.0, 0.0, 1.0), // straight up
            spread_deg: 40.0,
            size: 1.0,
            growth: 1.2,
            rate: 14.0,
            burst: None,
            speed: 0.4,
            speed_spread: 0.15,
            path_jitter: 0.0,
            seed: 0xC0FF_EE15_600D_5EED,
            swirl: 0.0,
            lifetime: 3.5,
            lifetime_spread: 0.5,
            accel: Vec3::new(0.0, 0.0, -0.02), // barely any gravity — a mist
            max_particles: 256,
            blend: EmitterBlend::KeepBg,
            synthesized: false,
            motion: MotionKind::Ballistic,
            lifetime_dist: None,
            speed_dist: None,
            size_dist: None,
            randomize: 1.0,
            color_jitter: None,
            color_ramp: None,
            glyph_pick: GlyphPick::Sequential,
            glyph_source: GlyphSource::Family,
            trail: None,
            bounce: None,
            visibility: DustVisibility::Always,
        }
    }
}

impl EmitterSpec {
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /// Start from the sane defaults (a gentle upward haze) and apply builder
    /// overrides incrementally. Equivalent to `EmitterSpec::default()` followed
    /// by `with_*` calls, but slightly more discoverable for the `re_level` parser
    /// which sets fields one key at a time.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Builder / `with_*` helpers
    // -----------------------------------------------------------------------

    /// Override [`kind`](Self::kind).
    /// `.lvl`: `kind`
    #[must_use]
    pub fn with_kind(mut self, kind: EmitterKind) -> Self {
        self.kind = kind;
        self
    }

    /// Override [`origin`](Self::origin).
    /// `.lvl`: `origin_world` / `origin_screen`
    #[must_use]
    pub fn with_origin(mut self, origin: EmitterOrigin) -> Self {
        self.origin = origin;
        self
    }

    /// Override the three colour stops.
    /// `.lvl`: `colors`
    #[must_use]
    pub fn with_colors(mut self, c0: u32, c1: u32, c2: u32) -> Self {
        self.colors = [c0, c1, c2];
        self
    }

    /// Override [`color_mode`](Self::color_mode).
    /// `.lvl`: `color_mode`
    #[must_use]
    pub fn with_color_mode(mut self, mode: ColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    /// Override [`family`](Self::family) by name.
    /// `.lvl`: `family`
    #[must_use]
    pub fn with_family(mut self, name: impl Into<String>) -> Self {
        self.family = name.into();
        self
    }

    /// Override the explicit [`glyph_set`](Self::glyph_set) escape hatch.
    /// `.lvl`: `glyph_set`
    #[must_use]
    pub fn with_glyph_set(mut self, glyphs: Vec<char>) -> Self {
        self.glyph_set = Some(glyphs);
        self
    }

    /// Override [`bold`](Self::bold).
    /// `.lvl`: `bold`
    #[must_use]
    pub fn with_bold(mut self, bold: BoldMode) -> Self {
        self.bold = bold;
        self
    }

    /// Override the emission direction and cone spread.
    /// `.lvl`: `direction` + `spread_deg`
    ///
    /// `spread_deg` is clamped to `[0, 180]` immediately.
    #[must_use]
    pub fn with_direction(mut self, dir: Vec3, spread_deg: f32) -> Self {
        self.direction = dir;
        self.spread_deg = spread_deg.clamp(0.0, 180.0);
        self
    }

    /// Override [`size`](Self::size) and [`growth`](Self::growth).
    /// `.lvl`: `size` + `growth`
    #[must_use]
    pub fn with_size(mut self, size: f32, growth: f32) -> Self {
        self.size = size;
        self.growth = growth;
        self
    }

    /// Override [`rate`](Self::rate).
    /// `.lvl`: `rate`
    #[must_use]
    pub fn with_rate(mut self, rate: f32) -> Self {
        self.rate = rate;
        self
    }

    /// Set a one-shot [`burst`](Self::burst) count (disables continuous streaming).
    /// `.lvl`: `burst`
    #[must_use]
    pub fn with_burst(mut self, count: u32) -> Self {
        self.burst = Some(count);
        self
    }

    /// Override [`speed`](Self::speed) and [`speed_spread`](Self::speed_spread).
    /// `.lvl`: `speed` + `speed_spread`
    #[must_use]
    pub fn with_speed(mut self, speed: f32, speed_spread: f32) -> Self {
        self.speed = speed;
        self.speed_spread = speed_spread;
        self
    }

    /// Override path randomisation parameters.
    /// `.lvl`: `path_jitter` + `seed` + `swirl`
    #[must_use]
    pub fn with_path_rng(mut self, path_jitter: f32, seed: u64, swirl: f32) -> Self {
        self.path_jitter = path_jitter;
        self.seed = seed;
        self.swirl = swirl;
        self
    }

    /// Override [`lifetime`](Self::lifetime) and [`lifetime_spread`](Self::lifetime_spread).
    /// `.lvl`: `lifetime` + `lifetime_spread`
    #[must_use]
    pub fn with_lifetime(mut self, lifetime: f32, lifetime_spread: f32) -> Self {
        self.lifetime = lifetime;
        self.lifetime_spread = lifetime_spread;
        self
    }

    /// Override [`accel`](Self::accel).
    /// `.lvl`: `accel`
    #[must_use]
    pub fn with_accel(mut self, accel: Vec3) -> Self {
        self.accel = accel;
        self
    }

    /// Override [`max_particles`](Self::max_particles).
    /// `.lvl`: `max_particles`
    ///
    /// Silently clamps to `1` if `count == 0`.
    #[must_use]
    pub fn with_max_particles(mut self, count: u32) -> Self {
        self.max_particles = count.max(1);
        self
    }

    /// Override [`blend`](Self::blend).
    /// `.lvl`: `blend`
    #[must_use]
    pub fn with_blend(mut self, blend: EmitterBlend) -> Self {
        self.blend = blend;
        self
    }

    /// Override [`motion`](Self::motion).
    /// `.lvl`: `motion`
    #[must_use]
    pub fn with_motion(mut self, motion: MotionKind) -> Self {
        self.motion = motion;
        self
    }

    /// Override the [`Distribution`] overrides for lifetime/speed/size.
    /// `.lvl`: the distribution grammar on `lifetime`/`speed`/`size`
    #[must_use]
    pub fn with_distributions(
        mut self,
        lifetime: Option<Distribution>,
        speed: Option<Distribution>,
        size: Option<Distribution>,
    ) -> Self {
        self.lifetime_dist = lifetime;
        self.speed_dist = speed;
        self.size_dist = size;
        self
    }

    /// Override [`randomize`](Self::randomize).
    /// `.lvl`: `randomize`
    #[must_use]
    pub fn with_randomize(mut self, randomize: f32) -> Self {
        self.randomize = randomize;
        self
    }

    /// Override [`color_jitter`](Self::color_jitter).
    /// `.lvl`: `color_jitter`
    #[must_use]
    pub fn with_color_jitter(mut self, jitter: ColorJitter) -> Self {
        self.color_jitter = Some(jitter);
        self
    }

    /// Override [`color_ramp`](Self::color_ramp) (2-8 stops).
    /// `.lvl`: `color_ramp`
    #[must_use]
    pub fn with_color_ramp(mut self, stops: Vec<u32>) -> Self {
        self.color_ramp = Some(stops);
        self
    }

    /// Override [`glyph_pick`](Self::glyph_pick).
    /// `.lvl`: `glyph_pick`
    #[must_use]
    pub fn with_glyph_pick(mut self, pick: GlyphPick) -> Self {
        self.glyph_pick = pick;
        self
    }

    /// Override [`glyph_source`](Self::glyph_source).
    /// `.lvl`: `glyph_source`
    #[must_use]
    pub fn with_glyph_source(mut self, source: GlyphSource) -> Self {
        self.glyph_source = source;
        self
    }

    /// Override [`trail`](Self::trail).
    /// `.lvl`: `trail`
    #[must_use]
    pub fn with_trail(mut self, trail: TrailSpec) -> Self {
        self.trail = Some(trail);
        self
    }

    /// Override [`bounce`](Self::bounce) (restitution coefficient).
    /// `.lvl`: `bounce`
    #[must_use]
    pub fn with_bounce(mut self, restitution: f32) -> Self {
        self.bounce = Some(restitution);
        self
    }

    /// Override [`visibility`](Self::visibility).
    /// `.lvl`: `visibility`
    #[must_use]
    pub fn with_visibility(mut self, visibility: DustVisibility) -> Self {
        self.visibility = visibility;
        self
    }

    // -----------------------------------------------------------------------
    // Validation / clamping
    // -----------------------------------------------------------------------

    /// Clamp / fix any out-of-range fields in-place. Returns `&mut self` for
    /// chaining. This is the canoncial "make it sensible before handing to the
    /// renderer" step; `re_render::Emitter::from_spec` should call this first.
    ///
    /// Rules applied:
    /// - `spread_deg` → clamped to `[0, 180]`
    /// - `lifetime` → at least `0.01` seconds
    /// - `lifetime_spread` → non-negative, capped at `lifetime - 0.01` so the
    ///   shortest possible particle life is always `≥ 0.01 s`
    /// - `max_particles` → at least `1`
    /// - `rate` → non-negative
    /// - `speed` → non-negative
    /// - `speed_spread` → non-negative
    /// - `path_jitter` → non-negative
    /// - `size` → at least `f32::EPSILON` (zero size is degenerate)
    /// - `growth` → at least `f32::EPSILON`
    pub fn validate(&mut self) {
        self.spread_deg = self.spread_deg.clamp(0.0, 180.0);
        self.lifetime = self.lifetime.max(0.01);
        self.lifetime_spread = self.lifetime_spread.abs().min(self.lifetime - 0.01);
        self.max_particles = self.max_particles.max(1);
        self.rate = self.rate.max(0.0);
        self.speed = self.speed.max(0.0);
        self.speed_spread = self.speed_spread.max(0.0);
        self.path_jitter = self.path_jitter.max(0.0);
        self.size = self.size.max(f32::EPSILON);
        self.growth = self.growth.max(f32::EPSILON);

        // ── variety fields ──────────────────────────────────────────────
        self.randomize = self.randomize.max(0.0);
        if let Some(j) = self.color_jitter.as_mut() {
            j.h_deg = j.h_deg.abs();
            j.s = j.s.abs().min(1.0);
            j.l = j.l.abs().min(1.0);
            j.dance = j.dance.clamp(0.0, 1.0);
        }
        if let Some(stops) = self.color_ramp.as_ref() {
            if stops.len() < 2 {
                self.color_ramp = None; // degenerate ramp — fall back to `colors`.
            }
        }
        if let Some(t) = self.trail.as_mut() {
            t.count = t.count.clamp(1, 6);
            t.fade = t.fade.clamp(0.0, 1.0);
        }
        if let Some(b) = self.bounce.as_mut() {
            *b = b.clamp(0.0, 1.0);
        }
        if let DustVisibility::Lit { gain } = &mut self.visibility {
            *gain = gain.max(0.0);
        }
        match &mut self.motion {
            MotionKind::Curl {
                scale, strength, ..
            } => {
                *scale = scale.max(f32::EPSILON);
                *strength = strength.max(0.0);
            }
            MotionKind::Attract { strength, .. } => *strength = strength.max(0.0),
            MotionKind::Lorenz { scale } => *scale = scale.max(f32::EPSILON),
            MotionKind::Flock { n, seek_light, .. } => {
                *n = (*n).clamp(1, 64);
                *seek_light = seek_light.max(0.0);
            }
            MotionKind::Ripple { speed } => *speed = speed.max(0.01),
            MotionKind::Ballistic => {}
        }
    }

    /// Return a validated clone. Non-mutating convenience for callers that
    /// prefer not to mutate in place.
    #[must_use]
    pub fn validated(mut self) -> Self {
        self.validate();
        self
    }

    /// True if this spec passes all range checks without modification. Useful
    /// for unit tests and `re_level` parse-time diagnostics.
    pub fn is_valid(&self) -> bool {
        (0.0..=180.0).contains(&self.spread_deg)
            && self.lifetime >= 0.01
            && self.lifetime_spread >= 0.0
            && self.lifetime_spread <= self.lifetime - 0.01
            && self.max_particles >= 1
            && self.rate >= 0.0
            && self.speed >= 0.0
            && self.speed_spread >= 0.0
            && self.path_jitter >= 0.0
            && self.size >= f32::EPSILON
            && self.growth >= f32::EPSILON
            && self.randomize >= 0.0
            && self.color_ramp.as_ref().is_none_or(|s| s.len() >= 2)
            && self
                .trail
                .as_ref()
                .is_none_or(|t| (1..=6).contains(&t.count))
            && self.bounce.as_ref().is_none_or(|b| (0.0..=1.0).contains(b))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default is sensible & valid ─────────────────────────────────────────

    #[test]
    fn default_is_valid() {
        let spec = EmitterSpec::default();
        assert!(
            spec.is_valid(),
            "default EmitterSpec must pass is_valid(); spec = {spec:?}"
        );
    }

    #[test]
    fn default_has_sensible_values() {
        let spec = EmitterSpec::default();
        // A gentle haze: should be world-anchored.
        assert!(
            matches!(spec.origin, EmitterOrigin::World(_)),
            "default origin must be World"
        );
        // Should have a non-zero rate so it actually emits particles.
        assert!(spec.rate > 0.0, "default rate must be positive");
        // Should have a positive lifetime.
        assert!(spec.lifetime > 0.0, "default lifetime must be positive");
        // Should have a non-zero max_particles cap.
        assert!(
            spec.max_particles >= 1,
            "default max_particles must be >= 1"
        );
        // Family must be non-empty (a known name).
        assert!(!spec.family.is_empty(), "default family must be non-empty");
        // The default blend is KeepBg (transparent splat).
        assert_eq!(spec.blend, EmitterBlend::KeepBg);
        // Default direction is upward.
        assert!(spec.direction.z > 0.0, "default direction should be upward");
    }

    // ── Builder / with_* round-trips ─────────────────────────────────────────

    #[test]
    fn builder_with_origin_world() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let spec = EmitterSpec::new().with_origin(EmitterOrigin::World(pos));
        assert_eq!(spec.origin, EmitterOrigin::World(pos));
    }

    #[test]
    fn builder_with_origin_screen() {
        let spec = EmitterSpec::new().with_origin(EmitterOrigin::Screen {
            col: 10.5,
            row: 5.0,
        });
        assert!(
            matches!(spec.origin, EmitterOrigin::Screen { col, row } if col == 10.5 && row == 5.0)
        );
    }

    #[test]
    fn builder_with_colors_roundtrip() {
        let spec = EmitterSpec::new().with_colors(0xFF0000, 0x00FF00, 0x0000FF);
        assert_eq!(spec.colors, [0xFF0000, 0x00FF00, 0x0000FF]);
    }

    #[test]
    fn builder_with_color_mode() {
        let spec = EmitterSpec::new().with_color_mode(ColorMode::RandomPick);
        assert_eq!(spec.color_mode, ColorMode::RandomPick);
    }

    #[test]
    fn builder_with_family() {
        let spec = EmitterSpec::new().with_family("growth_bars");
        assert_eq!(spec.family, "growth_bars");
    }

    #[test]
    fn builder_with_glyph_set() {
        let glyphs = vec!['.', 'o', 'O', '@'];
        let spec = EmitterSpec::new().with_glyph_set(glyphs.clone());
        assert_eq!(spec.glyph_set, Some(glyphs));
    }

    #[test]
    fn builder_with_bold() {
        let spec = EmitterSpec::new().with_bold(BoldMode::ByProximity);
        assert_eq!(spec.bold, BoldMode::ByProximity);
    }

    #[test]
    fn builder_with_direction_clamps_spread() {
        // spread_deg > 180 must be clamped immediately.
        let spec = EmitterSpec::new().with_direction(Vec3::new(1.0, 0.0, 0.0), 999.0);
        assert_eq!(spec.spread_deg, 180.0);
        // spread_deg < 0 clamps to 0.
        let spec2 = EmitterSpec::new().with_direction(Vec3::Y, -10.0);
        assert_eq!(spec2.spread_deg, 0.0);
        // A valid spread passes through unchanged.
        let spec3 = EmitterSpec::new().with_direction(Vec3::Z, 45.0);
        assert_eq!(spec3.spread_deg, 45.0);
    }

    #[test]
    fn builder_with_size() {
        let spec = EmitterSpec::new().with_size(2.5, 3.0);
        assert_eq!(spec.size, 2.5);
        assert_eq!(spec.growth, 3.0);
    }

    #[test]
    fn builder_with_rate() {
        let spec = EmitterSpec::new().with_rate(100.0);
        assert_eq!(spec.rate, 100.0);
    }

    #[test]
    fn builder_with_burst() {
        let spec = EmitterSpec::new().with_burst(50);
        assert_eq!(spec.burst, Some(50));
    }

    #[test]
    fn builder_with_speed() {
        let spec = EmitterSpec::new().with_speed(2.0, 0.5);
        assert_eq!(spec.speed, 2.0);
        assert_eq!(spec.speed_spread, 0.5);
    }

    #[test]
    fn builder_with_path_rng() {
        let spec = EmitterSpec::new().with_path_rng(0.3, 12345, 0.8);
        assert_eq!(spec.path_jitter, 0.3);
        assert_eq!(spec.seed, 12345);
        assert_eq!(spec.swirl, 0.8);
    }

    #[test]
    fn builder_with_lifetime() {
        let spec = EmitterSpec::new().with_lifetime(5.0, 0.5);
        assert_eq!(spec.lifetime, 5.0);
        assert_eq!(spec.lifetime_spread, 0.5);
    }

    #[test]
    fn builder_with_accel() {
        let a = Vec3::new(0.0, 0.0, -9.8);
        let spec = EmitterSpec::new().with_accel(a);
        assert_eq!(spec.accel, a);
    }

    #[test]
    fn builder_with_max_particles_clamps_zero() {
        let spec = EmitterSpec::new().with_max_particles(0);
        assert_eq!(spec.max_particles, 1, "max_particles=0 must clamp to 1");
    }

    #[test]
    fn builder_with_max_particles_roundtrip() {
        let spec = EmitterSpec::new().with_max_particles(512);
        assert_eq!(spec.max_particles, 512);
    }

    #[test]
    fn builder_with_blend() {
        let spec = EmitterSpec::new().with_blend(EmitterBlend::Opaque);
        assert_eq!(spec.blend, EmitterBlend::Opaque);
    }

    // ── Chained builder produces valid spec ──────────────────────────────────

    #[test]
    fn builder_chain_is_valid() {
        let spec = EmitterSpec::new()
            .with_origin(EmitterOrigin::World(Vec3::new(5.0, 5.0, 1.5)))
            .with_colors(0xFFD27F, 0xFF7A1A, 0x6E1200)
            .with_color_mode(ColorMode::RampOverAge)
            .with_family("growth_dot")
            .with_bold(BoldMode::ByAge)
            .with_direction(Vec3::new(0.0, 0.0, 1.0), 30.0)
            .with_size(1.0, 1.5)
            .with_rate(24.0)
            .with_speed(0.9, 0.2)
            .with_path_rng(0.0, 42, 0.0)
            .with_lifetime(2.2, 0.3)
            .with_accel(Vec3::new(0.0, 0.0, 0.35))
            .with_max_particles(256)
            .with_blend(EmitterBlend::KeepBg)
            .validated();
        assert!(spec.is_valid(), "chained builder result must be valid");
        assert_eq!(spec.family, "growth_dot");
        assert_eq!(spec.colors[0], 0xFFD27F);
        assert_eq!(spec.max_particles, 256);
    }

    // ── Validate / clamp ────────────────────────────────────────────────────

    #[test]
    fn validate_clamps_spread_deg() {
        let mut spec = EmitterSpec {
            spread_deg: -10.0,
            ..Default::default()
        };
        spec.validate();
        assert_eq!(spec.spread_deg, 0.0, "negative spread_deg must clamp to 0");

        spec.spread_deg = 250.0;
        spec.validate();
        assert_eq!(spec.spread_deg, 180.0, "spread_deg > 180 must clamp to 180");
    }

    #[test]
    fn validate_clamps_lifetime() {
        let mut spec = EmitterSpec {
            lifetime: 0.0,
            ..Default::default()
        };
        spec.validate();
        assert!(
            spec.lifetime >= 0.01,
            "lifetime must be >= 0.01 after validate"
        );
    }

    #[test]
    fn validate_clamps_lifetime_spread_to_safe_range() {
        // lifetime_spread must never make the shortest particle life < 0.01.
        let mut spec = EmitterSpec {
            lifetime: 1.0,
            lifetime_spread: 5.0,
            ..Default::default()
        }; // way too large
        spec.validate();
        assert!(
            spec.lifetime_spread <= spec.lifetime - 0.01,
            "lifetime_spread={} must be <= lifetime-0.01={}",
            spec.lifetime_spread,
            spec.lifetime - 0.01
        );
    }

    #[test]
    fn validate_clamps_max_particles_to_one() {
        let mut spec = EmitterSpec {
            max_particles: 0,
            ..Default::default()
        };
        spec.validate();
        assert_eq!(spec.max_particles, 1);
    }

    #[test]
    fn validate_clamps_negative_rate_and_speed() {
        let mut spec = EmitterSpec {
            rate: -50.0,
            speed: -3.0,
            speed_spread: -1.0,
            path_jitter: -0.5,
            ..Default::default()
        };
        spec.validate();
        assert_eq!(spec.rate, 0.0);
        assert_eq!(spec.speed, 0.0);
        assert_eq!(spec.speed_spread, 0.0);
        assert_eq!(spec.path_jitter, 0.0);
    }

    #[test]
    fn validate_clamps_degenerate_size_and_growth() {
        let mut spec = EmitterSpec {
            size: 0.0,
            growth: -1.0,
            ..Default::default()
        };
        spec.validate();
        assert!(spec.size >= f32::EPSILON);
        assert!(spec.growth >= f32::EPSILON);
    }

    #[test]
    fn is_valid_detects_bad_spread() {
        let mut spec = EmitterSpec {
            spread_deg: 200.0,
            ..Default::default()
        };
        assert!(!spec.is_valid(), "spread_deg=200 must fail is_valid");
        spec.validate();
        assert!(spec.is_valid(), "after validate, must be valid");
    }

    // ── PartialEq ───────────────────────────────────────────────────────────

    #[test]
    fn partial_eq_same_default() {
        let a = EmitterSpec::default();
        let b = EmitterSpec::default();
        assert_eq!(a, b, "two defaults must be equal");
    }

    #[test]
    fn partial_eq_different_family() {
        let a = EmitterSpec::new().with_family("haze_o");
        let b = EmitterSpec::new().with_family("growth_bars");
        assert_ne!(a, b, "different family names must not be equal");
    }

    #[test]
    fn partial_eq_different_origin() {
        let a = EmitterSpec::new().with_origin(EmitterOrigin::World(Vec3::ZERO));
        let b = EmitterSpec::new().with_origin(EmitterOrigin::Screen { col: 0.0, row: 0.0 });
        assert_ne!(a, b, "World vs Screen origin must not be equal");
    }

    // ── Clone ────────────────────────────────────────────────────────────────

    #[test]
    fn clone_is_equal_to_original() {
        let original = EmitterSpec::new()
            .with_family("anim_spin")
            .with_colors(0x001122, 0x334455, 0x667788)
            .with_burst(10);
        let cloned = original.clone();
        assert_eq!(original, cloned, "clone must equal original");
    }

    // ── Enum defaults ────────────────────────────────────────────────────────

    #[test]
    fn color_mode_default_is_ramp_over_age() {
        assert_eq!(ColorMode::default(), ColorMode::RampOverAge);
    }

    #[test]
    fn bold_mode_default_is_off() {
        assert_eq!(BoldMode::default(), BoldMode::Off);
    }

    #[test]
    fn emitter_blend_default_is_keep_bg() {
        assert_eq!(EmitterBlend::default(), EmitterBlend::KeepBg);
    }

    #[test]
    fn default_is_not_synthesized() {
        // Every hand-built/authored spec (including one an editor `push`es)
        // must default to `synthesized = false`; only `re_level`'s snow
        // expansion sets it `true`, and it does so explicitly on each spec.
        assert!(!EmitterSpec::default().synthesized);
        assert!(!EmitterSpec::new().synthesized);
    }

    // ── Variety fields ───────────────────────────────────────────────────────

    #[test]
    fn default_variety_fields_are_neutral() {
        let spec = EmitterSpec::default();
        assert_eq!(spec.motion, MotionKind::Ballistic);
        assert_eq!(spec.lifetime_dist, None);
        assert_eq!(spec.speed_dist, None);
        assert_eq!(spec.size_dist, None);
        assert_eq!(spec.randomize, 1.0);
        assert_eq!(spec.color_jitter, None);
        assert_eq!(spec.color_ramp, None);
        assert_eq!(spec.glyph_pick, GlyphPick::Sequential);
        assert_eq!(spec.glyph_source, GlyphSource::Family);
        assert_eq!(spec.trail, None);
        assert_eq!(spec.bounce, None);
        assert_eq!(spec.visibility, DustVisibility::Always);
    }

    #[test]
    fn distribution_uniform_spans_range() {
        let d = Distribution::Uniform { lo: 2.0, hi: 6.0 };
        assert_eq!(d.sample(0.0, 0.0), 2.0);
        assert!((d.sample(1.0, 0.0) - 6.0).abs() < 1e-4);
        let mid = d.sample(0.5, 0.0);
        assert!((3.9..=4.1).contains(&mid));
    }

    #[test]
    fn distribution_gaussian_clamped_at_3_sigma() {
        let d = Distribution::Gaussian {
            mean: 10.0,
            std: 2.0,
        };
        // Sweep many draws; none may exceed mean ± 3*std.
        for i in 0..200 {
            let u0 = ((i * 37) % 997) as f32 / 997.0;
            let u1 = ((i * 53 + 11) % 991) as f32 / 991.0;
            let v = d.sample(u0.max(1e-4), u1);
            assert!(
                (10.0 - 6.0..=10.0 + 6.0).contains(&v),
                "gaussian sample {v} escaped the 3-sigma clamp"
            );
        }
    }

    #[test]
    fn distribution_power_law_endpoints() {
        let d = Distribution::PowerLaw {
            lo: 1.0,
            hi: 4.0,
            k: 3.0,
        };
        assert!((d.sample(0.0, 0.0) - 1.0).abs() < 1e-5);
        assert!((d.sample(1.0, 0.0) - 4.0).abs() < 1e-4);
        // k>1 biases toward `lo`: the midpoint draw stays below the linear
        // midpoint (2.5).
        assert!(d.sample(0.5, 0.0) < 2.5);
    }

    #[test]
    fn validate_clamps_variety_fields() {
        let mut spec = EmitterSpec {
            randomize: -3.0,
            color_jitter: Some(ColorJitter {
                h_deg: -40.0,
                s: 2.0,
                l: -2.0,
                dance: 5.0,
            }),
            trail: Some(TrailSpec {
                count: 200,
                fade: 5.0,
                dir_glyphs: true,
            }),
            bounce: Some(3.0),
            visibility: DustVisibility::Lit { gain: -1.0 },
            ..Default::default()
        };
        spec.validate();
        assert_eq!(spec.randomize, 0.0);
        let j = spec.color_jitter.unwrap();
        assert_eq!(j.h_deg, 40.0);
        assert_eq!(j.s, 1.0);
        assert_eq!(j.l, 1.0);
        assert_eq!(j.dance, 1.0);
        let t = spec.trail.unwrap();
        assert_eq!(t.count, 6);
        assert_eq!(t.fade, 1.0);
        assert_eq!(spec.bounce, Some(1.0));
        assert!(matches!(spec.visibility, DustVisibility::Lit { gain } if gain == 0.0));
    }

    #[test]
    fn validate_drops_degenerate_color_ramp() {
        let mut spec = EmitterSpec {
            color_ramp: Some(vec![0xFF0000]),
            ..Default::default()
        };
        spec.validate();
        assert_eq!(
            spec.color_ramp, None,
            "a single-stop ramp is degenerate and must be dropped"
        );
    }

    #[test]
    fn builder_with_motion_roundtrip() {
        let spec = EmitterSpec::new().with_motion(MotionKind::Curl {
            scale: 0.5,
            strength: 0.8,
            evolve: 0.2,
        });
        assert_eq!(
            spec.motion,
            MotionKind::Curl {
                scale: 0.5,
                strength: 0.8,
                evolve: 0.2
            }
        );
    }

    #[test]
    fn builder_with_trail_and_bounce() {
        let spec = EmitterSpec::new()
            .with_trail(TrailSpec {
                count: 5,
                fade: 0.4,
                dir_glyphs: false,
            })
            .with_bounce(0.35);
        assert_eq!(spec.trail.unwrap().count, 5);
        assert_eq!(spec.bounce, Some(0.35));
    }

    #[test]
    fn is_valid_rejects_bad_variety_fields_and_validate_fixes_them() {
        let mut spec = EmitterSpec {
            randomize: -1.0,
            ..Default::default()
        };
        assert!(!spec.is_valid());
        spec.validate();
        assert!(spec.is_valid());
    }

    #[test]
    fn emitter_kind_default_is_standard() {
        assert_eq!(EmitterKind::default(), EmitterKind::Standard);
        // The spec default must also be Standard so existing levels are unchanged.
        assert_eq!(EmitterSpec::default().kind, EmitterKind::Standard);
    }

    #[test]
    fn builder_with_kind_roundtrip() {
        let spec = EmitterSpec::new().with_kind(EmitterKind::GodRays);
        assert_eq!(spec.kind, EmitterKind::GodRays);
    }

    // ── Family name contract ─────────────────────────────────────────────────

    /// Verify the canonical family names the spec lists are non-empty strings.
    /// (Actual family lookup lives in re_render — we only store the name here.)
    #[test]
    fn canonical_family_names_are_valid_strings() {
        let names = [
            "haze_o",
            "haze_dot",
            "haze_wave",
            "haze_colon",
            "haze_bar",
            "growth_dot",
            "growth_density",
            "growth_bars",
            "anim_spin",
            "anim_arc",
            "anim_sweep",
            "lean_fill",
            "lean_stub",
        ];
        for name in names {
            assert!(!name.is_empty(), "family name must be non-empty: {name}");
            // Each name, when set, round-trips through the builder unchanged.
            let spec = EmitterSpec::new().with_family(name);
            assert_eq!(spec.family, name, "family name must round-trip: {name}");
        }
    }
}
