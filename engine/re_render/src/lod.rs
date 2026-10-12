//! Unified distance Level-of-Detail (LOD) — ONE engine-configurable near / mid / far
//! tier system that replaces the formerly-scattered, hard-cutoff distance bands.
//!
//! ## Why this module exists
//!
//! Before this module the per-cell shade DETAIL was gated by several INDEPENDENT
//! hard distance comparisons, all keyed on the same two magic literals (6 and 13
//! world cells):
//!
//! * the SHADE band ([`crate::ascii_shader::ShadeLod`]): relief gradient dropped at
//!   `6`, noise octaves capped at `13`;
//! * the SHADOW band ([`crate::shadow::shadow_key_for_lod`]): full soft `<6`, single
//!   hard ray `6..13`, no ray `>13`.
//!
//! Each was a bare `distance < CONST` / `distance > CONST` test. STATUS flagged that
//! these HARD cutoffs at depth 6 / 13 visibly POP while the player walks (a cell
//! crossing exactly `13.0` snaps from "has a shadow ray" to "no shadow ray" between
//! two adjacent frames). They were also un-configurable — the thresholds lived as
//! `const`s in two different files.
//!
//! [`LodConfig`] folds those decisions into ONE place: three tiers
//! ([`LodTier::Near`] / [`LodTier::Mid`] / [`LodTier::Far`]) with **configurable
//! distance thresholds** AND a **Schmitt hysteresis deadband** per boundary so a
//! tier does not flip on a single-frame distance wobble. Each tier carries the
//! DETAIL FLAGS the old scattered bands set (relief gradient, noise-octave cap,
//! shadow quality) — so the shade path and the shadow path now ask the SAME tier
//! "what detail do I render?" instead of each re-deriving it from a raw distance.
//!
//! ## Determinism / golden-safety
//!
//! LOD is a **pure function of `(distance, config, material-override)`**. The active
//! [`LodConfig`] is set ONCE per cast via [`apply`](LodConfig::apply) (a thread-local, exactly the
//! pattern [`crate::config`] uses for the shading knobs — never an env read in the
//! hot path). Within a frame the config is constant, so the tier a cell resolves to
//! is reproducible → golden hashes hold.
//!
//! The hysteresis deadband is expressed as a value the LOD resolver can apply
//! WITHOUT per-frame state: a boundary has an `enter` (step finer) and a `leave`
//! (step coarser) distance, and the resolver is biased to the COARSER tier inside
//! the deadband. The existing temporal hysteresis pass (the cross-frame glyph
//! deadband in [`crate::ascii_shader`]) is unchanged and remains the single stateful
//! stage; this deadband is a static geometric gap, not a frame memory, so it stays
//! pure. (See [`LodConfig::tier_for_prev`] for the optional stateful refinement the
//! present-layer diff cache can drive — opt-in, never required for correctness.)
//!
//! ## Default == today
//!
//! [`LodConfig::default`] reproduces the historical thresholds EXACTLY
//! (`near_end = 6.0`, `far_start = 13.0`) with the per-tier detail flags set to what
//! the old bands did, so the default is **not** a visual regression. The new value
//! is: it is now (a) configurable via `TS_LOD_*` + the struct, (b) hysteresis-smoothed
//! so boundaries no longer pop, and (c) overridable per material so a hero object can
//! hold full detail farther out than the global curve.

use std::cell::Cell as StdCell;

use bevy::prelude::Resource;

// ── historical defaults (the old scattered literals, now in ONE place) ───────────

/// Default near→mid boundary (world cells). Historically `SHADOW_SOFT_DIST` /
/// `SHADE_RELIEF_DIST` = `6.0`: below it the full soft shadow + relief gradient ran.
pub const LOD_DEFAULT_NEAR_END: f32 = 6.0;

/// Default mid→far boundary (world cells). Historically `SHADOW_HARD_DIST` /
/// `SHADE_OCTAVE_DIST` = `13.0`: above it the shadow ray was skipped and the noise
/// octaves were capped.
pub const LOD_DEFAULT_FAR_START: f32 = 13.0;

/// Default hysteresis half-width (world cells) applied symmetrically around each
/// tier boundary. A cell must travel `enter` past a boundary to step to the finer
/// tier and `leave` past it (the other way) to step to the coarser tier; in between
/// the resolver holds the COARSER tier. `0.75` cells ≈ a couple of slow walk frames
/// — wide enough to kill the single-frame pop the STATUS issue flagged, narrow enough
/// that the band still tracks the player. `0.0` reproduces the historical HARD cutoff.
pub const LOD_DEFAULT_HYSTERESIS: f32 = 0.75;

/// The far-tier noise-octave cap (historically `SHADE_FAR_OCTAVES = 1`): a far cell's
/// fBm/ridged noise collapses to a single low-frequency tap.
pub const LOD_DEFAULT_FAR_OCTAVES: u32 = 1;

// ── the three tiers ──────────────────────────────────────────────────────────

/// The three distance LOD tiers, coarsest→finest is Far→Mid→Near. A cell resolves to
/// exactly one tier from its perpendicular depth via [`LodConfig::tier_for`]; the
/// shade path and the shadow path both read the SAME tier so the picture degrades
/// coherently (no shade-band / shadow-band skew).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LodTier {
    /// FULL detail — the near foreground the player reads: relief gradient ON, no
    /// octave cap, full requested shadow quality. Byte-identical to the pre-LOD path.
    Near,
    /// REDUCED detail: relief gradient dropped (lit with the flat face normal), noise
    /// octaves still full, shadow degraded to a single hard ray.
    Mid,
    /// MINIMAL detail: relief gradient dropped, noise octaves capped to a cheap base
    /// sample, no shadow ray at all. A far cell occupies ~1 glyph; the dropped detail
    /// is sub-pixel.
    Far,
}

/// What a tier renders — the DETAIL FLAGS that the formerly-scattered bands set,
/// now centralised on the tier. Pure data; consumed by the shade + shadow paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierDetail {
    /// Keep the 2-extra-tap relief gradient (visible normal-map perturbation). Was
    /// `distance < SHADE_RELIEF_DIST`.
    pub relief_gradient: bool,
    /// Upper bound on procedural-noise octaves. `u32::MAX` ⇒ no cap (the recipe's own
    /// count wins). Was the `distance > SHADE_OCTAVE_DIST ? SHADE_FAR_OCTAVES : MAX`
    /// decision.
    pub max_octaves: u32,
    /// Cast-shadow quality CEILING for this tier (the effective quality is the
    /// per-cast requested quality clamped DOWN to this). `Soft` near, `Hard` mid,
    /// `Off` far. Was the `shadow_key_for_lod` distance ladder.
    pub shadow: ShadowTierQuality,
}

/// The shadow-quality ceiling a tier permits. Ordered coarsest→finest as a small
/// enum so a tier can clamp the requested [`crate::ascii_shader::ShadowQuality`]
/// DOWN (a tier never raises shadow quality above what the cast asked for).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowTierQuality {
    /// No shadow ray (far). `shadow_key == 1.0`.
    Off,
    /// At most a single hard ray (mid).
    Hard,
    /// Up to the full requested soft-tap quality (near).
    Soft,
}

impl LodTier {
    /// The detail flags for this tier under `cfg` (the octave cap is read from the
    /// config so it is tunable). Centralises the old per-band decisions.
    #[inline]
    pub fn detail(self, cfg: &LodConfig) -> TierDetail {
        match self {
            LodTier::Near => TierDetail {
                relief_gradient: true,
                max_octaves: u32::MAX,
                shadow: ShadowTierQuality::Soft,
            },
            LodTier::Mid => TierDetail {
                relief_gradient: false,
                max_octaves: u32::MAX,
                shadow: ShadowTierQuality::Hard,
            },
            LodTier::Far => TierDetail {
                relief_gradient: false,
                max_octaves: cfg.far_octaves,
                shadow: ShadowTierQuality::Off,
            },
        }
    }
}

// ── the config ─────────────────────────────────────────────────────────────────

/// The unified LOD configuration — a Bevy [`Resource`] (so it is testable without
/// env) seeded once from `TS_LOD_*` env vars at plugin build. Holds the two tier
/// boundaries, a per-boundary hysteresis deadband, and the far-octave cap.
///
/// All distances are WORLD CELLS, so they are resolution-independent and the
/// small-grid golden cameras (whose geometry is all well inside the NEAR tier) are
/// byte-identical under the default.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct LodConfig {
    /// Near→Mid boundary (world cells). At/below `near_end` a cell is at most one
    /// hysteresis step from NEAR; default [`LOD_DEFAULT_NEAR_END`].
    pub near_end: f32,
    /// Mid→Far boundary (world cells). Default [`LOD_DEFAULT_FAR_START`].
    pub far_start: f32,
    /// Hysteresis half-width (world cells) applied around BOTH boundaries. The
    /// resolver holds the coarser tier inside `±hysteresis` of a boundary unless the
    /// previous tier was already the finer one (see [`tier_for_prev`](Self::tier_for_prev)).
    /// `0.0` ⇒ a hard cutoff (the historical pop). Default [`LOD_DEFAULT_HYSTERESIS`].
    pub hysteresis: f32,
    /// Far-tier noise-octave cap. Default [`LOD_DEFAULT_FAR_OCTAVES`].
    pub far_octaves: u32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            near_end: LOD_DEFAULT_NEAR_END,
            far_start: LOD_DEFAULT_FAR_START,
            hysteresis: LOD_DEFAULT_HYSTERESIS,
            far_octaves: LOD_DEFAULT_FAR_OCTAVES,
        }
    }
}

impl LodConfig {
    /// A config that reproduces the historical HARD cutoffs (no hysteresis) — the
    /// pre-unification look, used by tests that pin the old byte-for-byte band edges.
    pub const LEGACY_HARD: Self = Self {
        near_end: LOD_DEFAULT_NEAR_END,
        far_start: LOD_DEFAULT_FAR_START,
        hysteresis: 0.0,
        far_octaves: LOD_DEFAULT_FAR_OCTAVES,
    };

    /// Resolve the LOD tier for a cell at perpendicular `distance` (world cells),
    /// applying an optional per-material [`LodOverride`]. STATELESS: inside a
    /// boundary's deadband it holds the COARSER tier (a deterministic, golden-safe
    /// tie-break). For the stateful, pop-free walking refinement use
    /// [`tier_for_prev`](Self::tier_for_prev).
    ///
    /// A non-finite distance (sky/miss) → [`LodTier::Far`] (minimal detail), matching
    /// the old "non-finite depth skips the shadow ray" behaviour.
    #[inline]
    pub fn tier_for(&self, distance: f32, ov: Option<LodOverride>) -> LodTier {
        self.tier_for_prev(distance, ov, None)
    }

    /// Resolve the tier with optional knowledge of the PREVIOUS frame's tier for this
    /// cell (the Schmitt trigger). With `prev = None` the deadband is resolved toward
    /// the coarser tier (the stateless, golden-safe default). With `prev = Some(t)`
    /// the cell only steps FINER when it crosses `boundary - hysteresis` and only
    /// steps COARSER when it crosses `boundary + hysteresis`, so a cell wobbling
    /// within the deadband HOLDS its previous tier → no per-frame pop while walking.
    ///
    /// This is the pure core; the previous-tier input (when supplied) comes from the
    /// present layer's diff cache, not from internal mutable state — so the shader
    /// stays a pure function and goldens that pass `prev = None` are unaffected.
    pub fn tier_for_prev(
        &self,
        distance: f32,
        ov: Option<LodOverride>,
        prev: Option<LodTier>,
    ) -> LodTier {
        if !distance.is_finite() {
            return LodTier::Far;
        }
        let (near_end, far_start, hyst) = self.effective(ov);
        let d = distance.max(0.0);

        // Each boundary B has a finer-edge (B - hyst) and a coarser-edge (B + hyst).
        // Below finer-edge ⇒ unambiguously the finer tier; above coarser-edge ⇒
        // unambiguously the coarser tier; inside the band the previous tier holds
        // (or, stateless, the coarser tier).
        let resolve = |b: f32, finer: LodTier, coarser: LodTier| -> Option<LodTier> {
            if d < b - hyst {
                Some(finer)
            } else if d > b + hyst {
                Some(coarser)
            } else {
                // Inside the deadband: hold prev if it is one of the two adjacent
                // tiers (the true Schmitt anti-pop). With NO previous tier (the
                // stateless / golden path) bias to the FINER tier — i.e. KEEP detail
                // inside the band. This is the golden-safe choice: it can only EXTEND
                // a tier's reach by `+hyst`, never SHRINK it below the historical
                // hard-cutoff distance, so a near cell that was full-detail under the
                // old hard cutoff stays full-detail (no detail-loss regression). The
                // small extra cost is confined to the deadband.
                match prev {
                    Some(p) if p == finer || p == coarser => Some(p),
                    _ => Some(finer),
                }
            }
        };

        // Near→Mid boundary first; if it resolves to the COARSER side, fall through to
        // the Mid→Far boundary. The two boundaries are clamped non-overlapping in
        // `effective`, so a single distance never satisfies both finer edges.
        match resolve(near_end, LodTier::Near, LodTier::Mid) {
            Some(LodTier::Near) => LodTier::Near,
            _ => resolve(far_start, LodTier::Mid, LodTier::Far).unwrap_or(LodTier::Far),
        }
    }

    /// The detail flags for a cell at `distance` under an optional override — the
    /// one call the shade + shadow paths use.
    #[inline]
    pub fn detail_for(&self, distance: f32, ov: Option<LodOverride>) -> TierDetail {
        self.tier_for(distance, ov).detail(self)
    }

    /// Apply a per-material [`LodOverride`] to the global thresholds, returning the
    /// EFFECTIVE `(near_end, far_start, hysteresis)`. The boundaries are kept ordered
    /// and the hysteresis is clamped so the two deadbands never overlap (which would
    /// make the tie-break ambiguous): `hyst ≤ (far_start − near_end) / 2`.
    ///
    /// `pub(crate)` (not `pub`) so [`crate::depth_contrast`] can widen a level-fog band
    /// by the SAME per-material override the LOD tier resolver uses — see
    /// `depth_contrast::band_for_fog_ov` — without duplicating the widening rule.
    #[inline]
    pub(crate) fn effective(&self, ov: Option<LodOverride>) -> (f32, f32, f32) {
        let (mut near_end, mut far_start, mut hyst) =
            (self.near_end, self.far_start, self.hysteresis);
        if let Some(o) = ov {
            if let Some(n) = o.near_end {
                near_end = n;
            }
            if let Some(f) = o.far_start {
                far_start = f;
            }
            if let Some(h) = o.hysteresis {
                hyst = h;
            }
            // "Hold full detail to distance D": push BOTH boundaries out to at least D
            // so the cell stays NEAR (and then MID) far past the global curve.
            if let Some(d) = o.full_detail_to {
                near_end = near_end.max(d);
                far_start = far_start.max(d + (self.far_start - self.near_end).max(1.0));
            }
        }
        // Keep ordered + give a minimum mid band so a boundary pair is well-formed.
        far_start = far_start.max(near_end + 1.0);
        let max_hyst = ((far_start - near_end) * 0.5).max(0.0);
        hyst = hyst.clamp(0.0, max_hyst);
        (near_end, far_start, hyst)
    }

    /// Seed a config from the `TS_LOD_*` env vars. Called ONCE at plugin build for
    /// dev convenience; unset vars keep the [`Default`]. The ONLY place the process
    /// env is consulted for LOD (mirrors [`crate::config::RenderConfig::from_env`]).
    ///
    /// * `TS_LOD_NEAR` — near→mid boundary, world cells (finite `> 0`).
    /// * `TS_LOD_FAR`  — mid→far boundary, world cells (finite `> near`).
    /// * `TS_LOD_HYST` — hysteresis half-width, world cells (finite `≥ 0`).
    /// * `TS_LOD_FAR_OCTAVES` — far-tier octave cap (`≥ 1`).
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(v) = env_f32("TS_LOD_NEAR") {
            if v > 0.0 {
                cfg.near_end = v;
            }
        }
        if let Some(v) = env_f32("TS_LOD_FAR") {
            if v > cfg.near_end {
                cfg.far_start = v;
            }
        }
        if let Some(v) = env_f32("TS_LOD_HYST") {
            if v >= 0.0 {
                cfg.hysteresis = v;
            }
        }
        if let Some(v) = env_f32("TS_LOD_FAR_OCTAVES") {
            if v >= 1.0 {
                cfg.far_octaves = v as u32;
            }
        }
        cfg
    }

    /// Install this config as the ACTIVE LOD config the deep shade dispatch reads back
    /// via [`active`]. An explicit write — no env. The cast applies the live
    /// [`LodConfig`] resource at its entry; tests apply a constructed one.
    #[inline]
    pub fn apply(&self) {
        ACTIVE_LOD.with(|c| c.set(*self));
    }
}

#[inline]
fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
}

// ── per-material override (the "hero object holds detail" lever) ─────────────────

/// A per-material LOD OVERRIDE — the "key object" lever. A material that carries
/// `Some(LodOverride)` resolves its tiers against THESE thresholds instead of the
/// global [`LodConfig`], so a hero object (a puzzle node, a sign, a focal mirror) can
/// keep full detail farther out than the global curve while ordinary walls follow the
/// global LOD. Every field is optional: a `None` field falls back to the global value,
/// so an override can tweak just one boundary.
///
/// `Copy` + tiny so a [`crate::ascii_shader::Material`] stays `Copy` and can be a
/// `static` (the hot path never heap-allocates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodOverride {
    /// Override the near→mid boundary (world cells). `None` ⇒ global.
    pub near_end: Option<f32>,
    /// Override the mid→far boundary (world cells). `None` ⇒ global.
    pub far_start: Option<f32>,
    /// Override the hysteresis half-width. `None` ⇒ global.
    pub hysteresis: Option<f32>,
    /// "Stay full-detail to distance D" — the common authoring intent. Pushes BOTH
    /// boundaries out so the cell holds NEAR (then MID) detail well past the global
    /// far-start. `None` ⇒ no extension.
    pub full_detail_to: Option<f32>,
}

impl LodOverride {
    /// A no-op override (all fields fall back to global) — equivalent to `None`, but
    /// usable as a base to tweak one field.
    pub const INHERIT: Self = Self {
        near_end: None,
        far_start: None,
        hysteresis: None,
        full_detail_to: None,
    };

    /// "Hold full detail out to `d` world cells" — the hero-object constructor.
    #[inline]
    pub const fn full_detail_to(d: f32) -> Self {
        Self {
            near_end: None,
            far_start: None,
            hysteresis: None,
            full_detail_to: Some(d),
        }
    }
}

// ── footprint (projected texel size) metric ────────────────────────────────────
//
// The existing tiers above are keyed on raw perpendicular DISTANCE, which is
// blind to three things a real projected footprint carries — terminal
// resolution (`cols`), field of view, and surface incidence (a grazing floor
// is "far" in footprint terms much sooner than a face-on wall at the same
// distance). [`Footprint`] computes the projected texel size ρ (pattern
// texels per screen cell) that other modules (sky, colour/cell effects) can
// consume for the same purpose.
//
// [`LodConfig::tier_for_rho`] is a SEPARATE resolver from [`LodConfig::tier_for`]
// — it does not replace the live distance-keyed path (`ShadeLod::for_ctx` and
// every existing caller are UNCHANGED), so nothing here can move a golden frame.
// It is calibrated so that, for a face-on surface rendered at the REFERENCE
// config ([`LOD_FOOTPRINT_REF_COLS`] columns, [`LOD_FOOTPRINT_REF_FOV_DEG`]
// horizontal FOV — "100×32 @ FOV 60"),
// `tier_for_rho(Footprint::reference(d).rho, ov)` resolves to EXACTLY
// the same tier as `tier_for(d, ov)` for every `d` — proved by
// `footprint_tier_matches_distance_tier_at_the_reference_config`. That equivalence
// at the reference is the acceptance test: it is what makes ρ a drop-in, more
// accurate replacement for distance at OTHER resolutions/incidences without
// having re-tuned the near/mid/far boundaries by hand.

/// Reference column count the ρ-tier calibration is pinned to: the golden test's
/// framebuffer width (`re_game/tests/golden_framebuffer.rs`, `COLS = 100`).
pub const LOD_FOOTPRINT_REF_COLS: f32 = 100.0;

/// Reference horizontal FOV (degrees) the ρ-tier calibration is pinned to
/// ("100×32 @ FOV 60"). `re_render` has no live dependency on the window or
/// the `Fov` resource (those
/// live in `re_game`/`re_present`), so a footprint metric usable in a pure unit
/// test here has to pick a documented reference rather than read a live value —
/// this constant, and [`LOD_FOOTPRINT_REF_COLS`], are that reference.
pub const LOD_FOOTPRINT_REF_FOV_DEG: f32 = 60.0;

/// A cell's projected texel footprint: how many world units of a pattern
/// ONE terminal cell covers along each of the surface's two local axes, and the
/// resulting texel DENSITY `rho` (pattern texels per cell, assuming a pattern
/// texel is one world unit — a recipe with a different natural texel size scales
/// `rho` by its own `scale` knob before comparing it to anything).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint {
    /// World units one screen cell covers along the surface's "along" axis.
    pub f_u: f32,
    /// World units one screen cell covers along the surface's "up" axis.
    pub f_v: f32,
    /// `max(f_u, f_v)` — the tighter axis is what aliases first, so the coarser
    /// of the two dominates the LOD decision.
    pub rho: f32,
}

impl Footprint {
    /// Build the footprint for a perpendicular `distance` (world cells), the
    /// surface incidence cosines along each local axis (`cos_u`/`cos_v`; `1.0` is
    /// face-on, smaller values are grazing), under a `cols`-wide cast with
    /// horizontal FOV `fov_h_deg`: `f = d·α / max(|cosθ|, 0.05)`, where `α` is
    /// one screen cell's angular width. Degenerate inputs (`cols`
    /// or `distance` non-positive) never divide by zero or return non-finite —
    /// they clamp to the same floor `light_surface`'s incidence guard uses.
    pub fn for_view(distance: f32, cos_u: f32, cos_v: f32, cols: f32, fov_h_deg: f32) -> Self {
        let alpha = (fov_h_deg.to_radians() / cols.max(1.0)).max(0.0);
        let d = distance.max(0.0);
        let f_u = d * alpha / cos_u.abs().max(0.05);
        let f_v = d * alpha / cos_v.abs().max(0.05);
        Footprint {
            f_u,
            f_v,
            rho: f_u.max(f_v),
        }
    }

    /// The footprint at the CALIBRATION reference: a face-on surface
    /// (`cos_u = cos_v = 1.0`) under [`LOD_FOOTPRINT_REF_COLS`] /
    /// [`LOD_FOOTPRINT_REF_FOV_DEG`]. This is the quantity the tier boundaries
    /// below are pinned against.
    #[inline]
    pub fn reference(distance: f32) -> Self {
        Self::for_view(
            distance,
            1.0,
            1.0,
            LOD_FOOTPRINT_REF_COLS,
            LOD_FOOTPRINT_REF_FOV_DEG,
        )
    }
}

impl LodConfig {
    /// The ρ-per-world-cell slope at the calibration reference: since a
    /// face-on surface's footprint is `d·alpha` (incidence divides out to 1), ρ is
    /// EXACTLY linear in distance at the reference config, so a distance boundary
    /// `d0` maps to the ρ boundary `d0 * REFERENCE_RHO_PER_CELL` with no
    /// approximation. That linearity is what makes the reference-config
    /// equivalence exact rather than approximate.
    #[inline]
    fn reference_rho_per_cell() -> f32 {
        LOD_FOOTPRINT_REF_FOV_DEG.to_radians() / LOD_FOOTPRINT_REF_COLS
    }

    /// The calibrated ρ boundaries: `(near_end, far_start)` re-expressed in
    /// ρ units at the reference config, honouring the same per-material
    /// [`LodOverride`] [`effective`](Self::effective) applies to the distance path.
    #[inline]
    pub fn footprint_boundaries(&self, ov: Option<LodOverride>) -> (f32, f32) {
        let (near_end, far_start, _) = self.effective(ov);
        let k = Self::reference_rho_per_cell();
        (near_end * k, far_start * k)
    }

    /// Resolve the LOD tier from a FOOTPRINT ρ (e.g. [`Footprint::rho`]) instead
    /// of a raw distance — same hysteresis/override shape as
    /// [`tier_for`](Self::tier_for), just expressed in ρ-space. A non-finite ρ resolves to
    /// [`LodTier::Far`], matching the distance path's non-finite handling.
    #[inline]
    pub fn tier_for_rho(&self, rho: f32, ov: Option<LodOverride>) -> LodTier {
        self.tier_for_rho_prev(rho, ov, None)
    }

    /// [`tier_for_rho`](Self::tier_for_rho) with the Schmitt `prev`-tier input
    /// (mirrors [`tier_for_prev`](Self::tier_for_prev)).
    pub fn tier_for_rho_prev(
        &self,
        rho: f32,
        ov: Option<LodOverride>,
        prev: Option<LodTier>,
    ) -> LodTier {
        if !rho.is_finite() {
            return LodTier::Far;
        }
        let (near_end, far_start, hyst) = self.effective(ov);
        let k = Self::reference_rho_per_cell();
        let (rho_near, rho_far, rho_hyst) = (near_end * k, far_start * k, hyst * k);
        let r = rho.max(0.0);

        let resolve = |b: f32, finer: LodTier, coarser: LodTier| -> Option<LodTier> {
            if r < b - rho_hyst {
                Some(finer)
            } else if r > b + rho_hyst {
                Some(coarser)
            } else {
                match prev {
                    Some(p) if p == finer || p == coarser => Some(p),
                    _ => Some(finer),
                }
            }
        };
        match resolve(rho_near, LodTier::Near, LodTier::Mid) {
            Some(LodTier::Near) => LodTier::Near,
            _ => resolve(rho_far, LodTier::Mid, LodTier::Far).unwrap_or(LodTier::Far),
        }
    }
}

// ── active LOD config (explicit per-cast, never env in the hot path) ─────────────
//
// Mirrors `config.rs`: the LOD config is read deep inside the `Material`/recipe trait
// dispatch where a plain argument cannot reach without a large API churn. Held in a
// thread-local set ONLY explicitly (by `LodConfig::apply`, from the live resource at
// the cast boundary, or directly by a test). NOT an env global — no `env::var` in the
// path, so a unit test sets the config directly and asserts the resulting tier with
// zero environment dependence. Stable within a cast ⇒ golden-safe.

thread_local! {
    static ACTIVE_LOD: StdCell<LodConfig> = const { StdCell::new(LodConfig {
        near_end: LOD_DEFAULT_NEAR_END,
        far_start: LOD_DEFAULT_FAR_START,
        hysteresis: LOD_DEFAULT_HYSTERESIS,
        far_octaves: LOD_DEFAULT_FAR_OCTAVES,
    }) };
}

/// The ACTIVE LOD config for the current cast/thread. Set via [`LodConfig::apply`];
/// defaults to [`LodConfig::default`]. Read by the shade + shadow paths.
#[inline]
pub fn active() -> LodConfig {
    ACTIVE_LOD.with(|c| c.get())
}

// The "turn test": textures must move WITH the wall as the camera turns, not
// stay glued to the screen. Lives in its own file so the anchor contract
// reads as one document, same convention as `ascii_shader.rs`'s
// `anim_tests.rs`. `#[path]` because this module is not a `mod.rs`.
#[cfg(test)]
#[path = "turn_test.rs"]
mod turn_test;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_reproduces_historical_thresholds() {
        let c = LodConfig::default();
        assert_eq!(
            c.near_end, 6.0,
            "near→mid default = historical SHADOW_SOFT_DIST"
        );
        assert_eq!(
            c.far_start, 13.0,
            "mid→far default = historical SHADOW_HARD_DIST"
        );
        assert_eq!(
            c.far_octaves, 1,
            "far octave cap = historical SHADE_FAR_OCTAVES"
        );
    }

    #[test]
    fn legacy_hard_has_no_deadband() {
        // With hysteresis 0 the boundaries are a HARD step — the pre-unification look
        // a golden can pin. A whisker on either side of 6.0 picks different tiers.
        let c = LodConfig::LEGACY_HARD;
        assert_eq!(c.tier_for(5.999, None), LodTier::Near);
        assert_eq!(c.tier_for(6.001, None), LodTier::Mid);
        assert_eq!(c.tier_for(12.999, None), LodTier::Mid);
        assert_eq!(c.tier_for(13.001, None), LodTier::Far);
    }

    #[test]
    fn tiers_are_ordered_near_mid_far() {
        let c = LodConfig::default();
        assert_eq!(c.tier_for(1.0, None), LodTier::Near);
        assert_eq!(c.tier_for(9.0, None), LodTier::Mid);
        assert_eq!(c.tier_for(20.0, None), LodTier::Far);
    }

    #[test]
    fn detail_reduces_near_to_far() {
        // Near richer than Far: relief ON→OFF, octaves uncapped→capped, shadow Soft→Off.
        let c = LodConfig::default();
        let near = c.detail_for(1.0, None);
        let mid = c.detail_for(9.0, None);
        let far = c.detail_for(20.0, None);

        assert!(near.relief_gradient && !mid.relief_gradient && !far.relief_gradient);
        assert_eq!(near.max_octaves, u32::MAX);
        assert_eq!(mid.max_octaves, u32::MAX);
        assert_eq!(far.max_octaves, c.far_octaves);
        assert_eq!(near.shadow, ShadowTierQuality::Soft);
        assert_eq!(mid.shadow, ShadowTierQuality::Hard);
        assert_eq!(far.shadow, ShadowTierQuality::Off);
    }

    #[test]
    fn hysteresis_removes_the_hard_cutoff_pop() {
        // The STATUS bug: a cell crossing exactly 6.0 (or 13.0) between two frames
        // snaps tiers. With a deadband + the previous tier known, a cell wobbling
        // WITHIN the band HOLDS its tier — no pop.
        let c = LodConfig::default(); // hysteresis 0.75
        let b = c.near_end; // 6.0, deadband [5.25, 6.75]

        // Stateless: inside the band resolves to the FINER tier (keep detail) — this
        // can only EXTEND the near reach, never shrink it below the historical cutoff.
        assert_eq!(
            c.tier_for(b + 0.1, None),
            LodTier::Near,
            "inside band → finer (keep detail)"
        );
        assert_eq!(
            c.tier_for(b - 0.8, None),
            LodTier::Near,
            "below finer edge → Near"
        );
        assert_eq!(
            c.tier_for(b + 0.8, None),
            LodTier::Mid,
            "above coarser edge → Mid"
        );

        // Schmitt: a cell that WAS Near and wobbles into the band STAYS Near (no pop).
        assert_eq!(
            c.tier_for_prev(b - 0.1, None, Some(LodTier::Near)),
            LodTier::Near,
            "was Near, still inside the enter-band → hold Near (no pop)"
        );
        // It only drops to Mid once it clears the coarser edge.
        assert_eq!(
            c.tier_for_prev(b + 0.8, None, Some(LodTier::Near)),
            LodTier::Mid,
            "cleared the leave-edge → now Mid"
        );
        // Conversely a Mid cell holds Mid inside the band (won't pop up to Near).
        assert_eq!(
            c.tier_for_prev(b + 0.1, None, Some(LodTier::Mid)),
            LodTier::Mid,
            "was Mid, inside band → hold Mid"
        );
    }

    #[test]
    fn boundary_has_an_enter_leave_gap_not_a_single_threshold() {
        // Concrete before/after: the enter (step-finer) distance and
        // the leave (step-coarser) distance differ by 2×hysteresis.
        let c = LodConfig::default();
        let (near_end, _far, hyst) = c.effective(None);
        assert!(hyst > 0.0, "default has a real deadband");
        let enter = near_end - hyst; // step finer (Mid→Near) here
        let leave = near_end + hyst; // step coarser (Near→Mid) here
        assert!(
            leave - enter >= 2.0 * hyst - 1e-6,
            "boundary is a GAP, not one threshold"
        );
        assert!(enter < near_end && leave > near_end);
    }

    #[test]
    fn material_override_holds_detail_farther_than_global() {
        // A hero object: stay full-detail (NEAR) out to distance 30 — well past the
        // global far-start (13). A default material at the same distance is FAR.
        let c = LodConfig::default();
        let hero = Some(LodOverride::full_detail_to(30.0));

        let d = 25.0;
        assert_eq!(
            c.tier_for(d, None),
            LodTier::Far,
            "default material is FAR at 25"
        );
        assert_eq!(c.tier_for(d, hero), LodTier::Near, "hero holds NEAR at 25");

        // And the DETAIL the hero gets is the rich one.
        let hero_detail = c.detail_for(d, hero);
        assert!(hero_detail.relief_gradient, "hero keeps relief at 25");
        assert_eq!(
            hero_detail.shadow,
            ShadowTierQuality::Soft,
            "hero keeps soft shadow at 25"
        );
    }

    #[test]
    fn partial_override_inherits_unset_fields() {
        let c = LodConfig::default();
        // Only push the near boundary out; far + hyst inherit the global.
        let ov = Some(LodOverride {
            near_end: Some(10.0),
            ..LodOverride::INHERIT
        });
        let (near_end, far_start, hyst) = c.effective(ov);
        assert_eq!(near_end, 10.0);
        assert_eq!(far_start, 13.0, "far inherited");
        assert_eq!(hyst, c.hysteresis, "hysteresis inherited");
    }

    #[test]
    fn non_finite_distance_is_far() {
        let c = LodConfig::default();
        assert_eq!(c.tier_for(f32::INFINITY, None), LodTier::Far);
        assert_eq!(c.tier_for(f32::NAN, None), LodTier::Far);
    }

    #[test]
    fn from_env_defaults_when_unset() {
        // No TS_LOD_* set in this test process ⇒ defaults. (Other tests must not set
        // these env vars; from_env is dev-seed only.)
        // We can't guarantee env isolation cross-test, so only assert the parse helper.
        assert_eq!(env_f32("TS_LOD_DEFINITELY_UNSET_KEY_XYZ"), None);
    }

    #[test]
    fn apply_threads_the_active_config_without_env() {
        LodConfig {
            near_end: 4.0,
            far_start: 10.0,
            hysteresis: 0.5,
            far_octaves: 2,
        }
        .apply();
        let a = active();
        assert_eq!(a.near_end, 4.0);
        assert_eq!(a.far_octaves, 2);
        // Restore default for sibling tests on this thread.
        LodConfig::default().apply();
        assert_eq!(active(), LodConfig::default());
    }

    // ── footprint (projected texel size) metric ─────────────────────────────────

    /// THE ACCEPTANCE TEST: at the calibration reference (100 cols, 60° hfov,
    /// face-on), resolving a tier from
    /// `Footprint::reference(d).rho` must give EXACTLY the same answer as resolving
    /// it from `d` directly, for every distance a real cast would ask about — near,
    /// at both boundaries (inside and outside the hysteresis band), and far. Since
    /// nothing wires `tier_for_rho` into the live per-cell path (`ShadeLod::for_ctx`
    /// still calls `tier_for`/`detail_for` unchanged), this equivalence is also the
    /// proof that introducing the footprint metric moves ZERO existing goldens.
    #[test]
    fn footprint_tier_matches_distance_tier_at_the_reference_config() {
        let c = LodConfig::default();
        for &d in &[
            0.0, 1.0, 4.0, 5.99, 6.0, 6.01, 6.5, 6.99, 7.0, 9.0, 12.99, 13.0, 13.01, 13.5, 20.0,
            50.0,
        ] {
            let rho = Footprint::reference(d).rho;
            assert_eq!(
                c.tier_for_rho(rho, None),
                c.tier_for(d, None),
                "d={d} rho={rho} must resolve to the same tier via both metrics"
            );
        }
        // And with a per-material override, which both paths honour identically.
        let ov = Some(LodOverride::full_detail_to(30.0));
        for &d in &[10.0, 25.0, 40.0] {
            let rho = Footprint::reference(d).rho;
            assert_eq!(
                c.tier_for_rho(rho, ov),
                c.tier_for(d, ov),
                "override path, d={d}"
            );
        }
    }

    #[test]
    fn footprint_reference_grows_linearly_with_distance() {
        // Face-on at the reference config: no incidence division, so rho == k*d
        // exactly (the property `footprint_boundaries`'s calibration relies on).
        let k = LodConfig::default().footprint_boundaries(None).0 / LOD_DEFAULT_NEAR_END;
        for &d in &[0.0, 3.0, 6.0, 13.0, 40.0] {
            let got = Footprint::reference(d).rho;
            assert!(
                (got - k * d).abs() < 1e-5,
                "rho({d}) = {got}, expected {}",
                k * d
            );
        }
    }

    /// A HIGHER-RESOLUTION terminal (more columns ⇒ a narrower angular slice per
    /// cell) must see a SMALLER footprint at the SAME real distance — i.e. it
    /// resolves finer LOD tiers farther out: a 322-column terminal
    /// has 3.2× the texel density per cell of a 100-column one at the same
    /// distance". This is exactly what a pure distance metric cannot express.
    #[test]
    fn higher_column_count_yields_a_smaller_footprint_at_the_same_distance() {
        let d = 10.0;
        let low_res = Footprint::for_view(d, 1.0, 1.0, 100.0, LOD_FOOTPRINT_REF_FOV_DEG).rho;
        let hi_res = Footprint::for_view(d, 1.0, 1.0, 322.0, LOD_FOOTPRINT_REF_FOV_DEG).rho;
        assert!(
            hi_res < low_res,
            "322 cols ({hi_res}) must show a finer footprint than 100 ({low_res})"
        );

        // And it changes the resolved TIER: at a distance where the 100-col
        // reference is already Mid/Far, the 322-col footprint can still be Near.
        let c = LodConfig::default();
        let d2 = 8.0; // Mid under the reference (100 cols); should read finer at 322.
        let rho_ref = Footprint::for_view(d2, 1.0, 1.0, 100.0, LOD_FOOTPRINT_REF_FOV_DEG).rho;
        let rho_hi = Footprint::for_view(d2, 1.0, 1.0, 322.0, LOD_FOOTPRINT_REF_FOV_DEG).rho;
        assert_eq!(
            c.tier_for_rho(rho_ref, None),
            LodTier::Mid,
            "fixture check: reference reads Mid at d=8"
        );
        assert_eq!(
            c.tier_for_rho(rho_hi, None),
            LodTier::Near,
            "a wider terminal keeps full detail at the SAME real distance the \
             reference has already dropped it"
        );
    }

    /// A GRAZING surface (small incidence cosine) must show a LARGER footprint at
    /// the same distance than a face-on one — a floor stretching away from the
    /// camera is "far" in footprint terms sooner than a wall directly ahead, which
    /// a pure distance metric also cannot express.
    #[test]
    fn grazing_incidence_yields_a_larger_footprint_than_face_on() {
        let d = 5.0;
        let face_on = Footprint::for_view(d, 1.0, 1.0, 100.0, 60.0).rho;
        let grazing = Footprint::for_view(d, 0.1, 1.0, 100.0, 60.0).rho;
        assert!(
            grazing > face_on,
            "grazing ({grazing}) must exceed face-on ({face_on})"
        );

        let c = LodConfig::default();
        assert_eq!(
            c.tier_for_rho(face_on, None),
            LodTier::Near,
            "face-on at d=5 is Near"
        );
        assert_eq!(
            c.tier_for_rho(grazing, None),
            LodTier::Far,
            "the SAME distance, grazing, has already fallen to Far"
        );
    }

    #[test]
    fn footprint_tier_is_hysteresis_stable_like_the_distance_path() {
        let c = LodConfig::default();
        let rho_near = c.footprint_boundaries(None).0;
        // Was Near, wobbles back inside the enter-band → holds Near (no pop).
        let just_inside = rho_near * 1.02; // a touch above the boundary, inside the deadband
        assert_eq!(
            c.tier_for_rho_prev(just_inside, None, Some(LodTier::Near)),
            LodTier::Near,
            "rho hysteresis must hold the previous tier inside the deadband"
        );
    }

    #[test]
    fn footprint_tier_non_finite_rho_is_far() {
        let c = LodConfig::default();
        assert_eq!(c.tier_for_rho(f32::INFINITY, None), LodTier::Far);
        assert_eq!(c.tier_for_rho(f32::NAN, None), LodTier::Far);
    }

    #[test]
    fn footprint_for_view_never_divides_by_zero() {
        // Degenerate cols/distance must not panic or go non-finite.
        let f = Footprint::for_view(-5.0, 0.0, 0.0, 0.0, 60.0);
        assert!(f.f_u.is_finite() && f.f_v.is_finite() && f.rho.is_finite());
    }
}
