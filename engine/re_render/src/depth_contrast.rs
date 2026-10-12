//! Universal depth-contrast colour policy — a per-cell, distance-driven CONTRAST and
//! soft-emergence pass applied in the final shade
//! ([`Material::apply_lighting`](crate::ascii_shader::Material::apply_lighting)),
//! AFTER fog and BEFORE the animation / glyph tracks.
//!
//! ## The look it produces
//!
//! Far surfaces should lose CONTRAST — their ink (the `fg`) is pulled toward their
//! own local background (`bg`) — so distant contour lines / glyphs read as faint,
//! close-to-background marks; near surfaces keep FULL contrast (crisp, dense ink).
//! GLYPH SELECTION IS NOT TOUCHED — only colour. This is a colour lever on top of the
//! existing fog / LOD / freeze infrastructure, not a new geometry or glyph pass.
//!
//! Three composable effects, all distance-driven and all pure colour:
//!
//! 1. **Contrast collapse** — interpolate a per-cell `contrast ∈ [far, near]` from the
//!    SAME LOD band the rest of the renderer uses ([`crate::lod::LodConfig`]'s
//!    `near_end` / `far_start`), then pull `fg` toward `bg` by `(1 - contrast)`.
//! 2. **Fade-to-sky** — beyond `far_start`, additionally blend BOTH `fg` and `bg`
//!    toward a background / sky reference so the surface dissolves into the backdrop
//!    with no pop ("soft emergence"). The reference is the fog tint when set, else a
//!    neutral far grey (documented below).
//! 3. **Volume jitter** — a tiny, DETERMINISTIC, hash-STABLE per-cell colour wobble
//!    keyed on the cell's WORLD sample coordinate, giving distant flat fields a touch
//!    of volume without flicker (identical every frame → survives the cross-frame
//!    cache, never animates).
//!
//! ## Shipped default: ON. `Default`/thread-local stay OFF for test safety.
//!
//! Two different "default" live here, deliberately:
//!
//! * [`DepthContrastConfig::default`] (and the `ACTIVE_DEPTH` thread-local's initial
//!   value) is `enabled = false` — a strict no-op, byte-for-byte the pre-feature
//!   renderer. This is the baseline the hundreds of unrelated colour-exact unit
//!   tests across `re_render` were written against; changing it would silently
//!   recolour every one of them. Tests that want the effect construct/`apply()` it
//!   explicitly, as they already did before this note.
//! * [`DepthContrastConfig::from_env`] — the ONLY path the shipped game and
//!   `--dump` actually go through ([`crate::composite::CompositeRenderPlugin`] /
//!   the dump render fn) — defaults to `enabled = true`. The distance-depth effect
//!   ("far objects less detailed and less bright") ships ON; `TS_DEPTH=0` restores
//!   the historical pre-feature look exactly (every field back at its `Default`,
//!   so the off-switch is byte-identical, not just "dimmer").
//!
//! Also owns the DETAIL half of the effect (`§ ramp-step collapse` below): far
//! cells lose not just contrast but GLYPH VARIETY, collapsing [`crate::ascii_shader`]'s
//! body ramp onto a handful of coarse buckets, using the SAME LOD tier machinery
//! ([`crate::lod::LodConfig`]) — and its hysteresis deadband — the rest of the
//! renderer already relies on for pop-free distance transitions.
//!
//! ## Determinism / golden-safety
//!
//! Exactly the [`crate::config::RenderConfig`] / [`crate::lod::LodConfig`] pattern: a
//! Bevy [`Resource`] seeded ONCE from env at plugin build, applied to a thread-local
//! per cast via [`apply`](DepthContrastConfig::apply), read in the hot path via
//! [`active`]. NO env read in the hot path. Every input (distance, fg, bg, world
//! coord) is a pure function input, so a cell's shade is reproducible within a frame
//! and the volume jitter is identical across frames (hash-stable) ⇒ golden hashes hold.

use std::cell::Cell as StdCell;

use bevy::prelude::Resource;

/// Neutral far reference colour used by `fade_to_sky` when the cast supplies no fog
/// tint (`FogParams::color == None`, the fade-to-black default). A mid-low grey
/// (`0x202428`) rather than pure black: a distant surface dissolving toward a soft
/// grey backdrop reads as "atmosphere / haze" instead of a hard silhouette punched
/// into a black void. Packed `0x00RRGGBB`.
pub const DEPTH_NEUTRAL_FAR_REF: u32 = 0x0020_2428;

/// Default far-contrast: at/after `far_start` the ink is pulled 85% toward the cell's
/// own background (contrast = 0.15). Distant lines therefore read as faint, near
/// their backdrop — the spec's "low contrast, close to background".
pub const DEPTH_DEFAULT_FAR_CONTRAST: f32 = 0.15;

/// Default per-cell volume jitter amplitude (±, as a fraction of full brightness).
/// Small enough to read as grain/volume, not noise.
pub const DEPTH_DEFAULT_VOLUME_JITTER: f32 = 0.06;

/// Default RAMP-DITHER amplitude (stability spec §1.2 / finding E3). The depth-ink
/// falloff quantises a continuous brightness onto the ~10-step [`crate::ascii_shader`]
/// body ramp, so equal-depth cells land on the same step and the far field breaks into
/// visible CONTOUR BANDS — a staircase of glyph changes marching across a flat wall.
///
/// The fix is the classic one: jitter the ramp INDEX by `±0.5 · amplitude` of a step
/// before quantising, keyed on the cell's WORLD sample coord (never screen, never
/// time), so a band boundary dissolves into a stable stipple instead of a hard line.
/// Because the key is world-anchored the dither is IDENTICAL every frame for a given
/// world point — it breaks up bands without adding one frame of shimmer, and it
/// survives the cross-frame texel cache unchanged.
///
/// `0.5` ⇒ ±¼ step: enough to dissolve a boundary (only cells within a quarter step
/// of a threshold can move, and they move by at most one ramp step), not enough to
/// scramble the ordering the relief depends on. `1.0` gives the full ±½ step (total
/// band break-up); `0.0` turns the dither OFF and is byte-identical to the
/// pre-feature renderer.
pub const DEPTH_DEFAULT_RAMP_DITHER: f32 = 0.5;

/// Default distance (world cells) at which `fade_to_sky` reaches FULL blend — the
/// surface is nearly dissolved into the backdrop there. Matches the renderer's
/// historical FAR-FREEZE distance ([`crate::ascii_shader::FAR_FREEZE_DIST`] = 32.0),
/// so the soft emergence completes right where the far field hard-freezes anyway.
pub const DEPTH_DEFAULT_SKY_BLEND_FULL_AT: f32 = crate::ascii_shader::FAR_FREEZE_DIST;

/// Universal depth-contrast configuration — a Bevy [`Resource`] (testable without
/// env) seeded once from `TS_DEPTH*` env vars at plugin build. DEFAULT OFF
/// (`enabled = false`) ⇒ byte-identical to the pre-feature renderer.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct DepthContrastConfig {
    /// Master switch. `false` (DEFAULT) ⇒ the shade pass is a strict no-op (the
    /// renderer is byte-identical to today). `true` ⇒ apply the contrast / sky /
    /// jitter policy. Env: `TS_DEPTH` (`1`/`true` ⇒ on, `0`/`false` ⇒ off).
    pub enabled: bool,
    /// Contrast at/below the LOD `near_end` boundary — `1.0` (DEFAULT) keeps near ink
    /// at FULL contrast (no pull toward bg). Env: `TS_DEPTH_NEAR_CONTRAST`.
    pub near_contrast: f32,
    /// Contrast at/after the LOD `far_start` boundary. `< 1.0` pulls far ink toward its
    /// bg by `(1 - far_contrast)`. DEFAULT [`DEPTH_DEFAULT_FAR_CONTRAST`] (0.15 ⇒ 85%
    /// pull). Env: `TS_DEPTH_FAR_CONTRAST`.
    pub far_contrast: f32,
    /// Beyond `far_start`, additionally blend fg AND bg toward the background/sky
    /// reference so the surface flows into the backdrop with no pop. `true` (DEFAULT).
    /// Env: `TS_DEPTH_FADE_SKY` (`0`/`false` ⇒ off).
    pub fade_to_sky: bool,
    /// Distance (world cells) at which `fade_to_sky` reaches FULL blend (surface nearly
    /// dissolved). DEFAULT [`DEPTH_DEFAULT_SKY_BLEND_FULL_AT`] (the far-freeze dist).
    /// Env: `TS_DEPTH_SKY_FULL_AT`.
    pub sky_blend_full_at: f32,
    /// Per-cell volume-jitter amplitude (±, fraction of brightness), hash-stable on the
    /// world coord. DEFAULT [`DEPTH_DEFAULT_VOLUME_JITTER`] (0.06). `0.0` ⇒ no jitter.
    /// Env: `TS_DEPTH_JITTER`.
    pub volume_jitter: f32,
    /// Depth-ink RAMP-DITHER amplitude in ramp steps (the jitter is `±0.5 · this`).
    /// DEFAULT [`DEPTH_DEFAULT_RAMP_DITHER`] (0.5 ⇒ ±¼ step). `0.0` ⇒ OFF
    /// (byte-identical to the pre-dither renderer). Env: `TS_RAMP_DITHER`.
    ///
    /// UNLIKE every other field here this one is NOT gated on
    /// [`enabled`](Self::enabled): banding is a property of the ramp quantisation
    /// itself, which runs in the default renderer, so the dither has to run there
    /// too. The rest of the depth-contrast policy stays off by default.
    pub ramp_dither: f32,
}

impl Default for DepthContrastConfig {
    fn default() -> Self {
        Self {
            // OFF by default ⇒ byte-identical to today.
            enabled: false,
            near_contrast: 1.0,
            far_contrast: DEPTH_DEFAULT_FAR_CONTRAST,
            fade_to_sky: true,
            sky_blend_full_at: DEPTH_DEFAULT_SKY_BLEND_FULL_AT,
            volume_jitter: DEPTH_DEFAULT_VOLUME_JITTER,
            ramp_dither: DEPTH_DEFAULT_RAMP_DITHER,
        }
    }
}

impl DepthContrastConfig {
    /// Seed a config from the `TS_DEPTH*` env vars. Called ONCE at plugin build for dev
    /// convenience; unset vars keep the field [`Default`] EXCEPT `enabled`, whose
    /// unset fallback here is `true` (see the module-level "Shipped default: ON"
    /// note) — the distance-depth effect ships on by default; `TS_DEPTH=0`
    /// explicitly restores the historical look. The ONLY place the process env is
    /// consulted for depth-contrast (mirrors
    /// [`crate::config::RenderConfig::from_env`] / [`crate::lod::LodConfig::from_env`]).
    ///
    /// * `TS_DEPTH` — master switch. Unset ⇒ **ON** (shipped default). `1`/`true` ⇒
    ///   on, `0`/`false` ⇒ off (byte-identical to the pre-feature renderer).
    /// * `TS_DEPTH_NEAR_CONTRAST` — near contrast, finite `≥ 0`.
    /// * `TS_DEPTH_FAR_CONTRAST` — far contrast, finite `≥ 0`.
    /// * `TS_DEPTH_FADE_SKY` — fade-to-sky (`0`/`false` ⇒ off, else on).
    /// * `TS_DEPTH_SKY_FULL_AT` — full-blend distance, finite `> 0`.
    /// * `TS_DEPTH_JITTER` — jitter amplitude, finite `≥ 0`.
    /// * `TS_RAMP_DITHER` — depth-ink ramp-dither amplitude, finite `≥ 0` (`0` ⇒ off).
    pub fn from_env() -> Self {
        // Shipped default: ON. `Self::default()` stays OFF (test-harness baseline —
        // see the module doc); only the env-seeded config flips the master switch,
        // so this is the ONE place the shipped/dev-run behaviour actually differs
        // from the unit-test baseline.
        let mut cfg = Self {
            enabled: true,
            ..Self::default()
        };
        if let Some(b) = env_bool("TS_DEPTH") {
            cfg.enabled = b;
        }
        if let Some(v) = env_f32("TS_DEPTH_NEAR_CONTRAST") {
            if v >= 0.0 {
                cfg.near_contrast = v;
            }
        }
        if let Some(v) = env_f32("TS_DEPTH_FAR_CONTRAST") {
            if v >= 0.0 {
                cfg.far_contrast = v;
            }
        }
        if let Some(b) = env_bool("TS_DEPTH_FADE_SKY") {
            cfg.fade_to_sky = b;
        }
        if let Some(v) = env_f32("TS_DEPTH_SKY_FULL_AT") {
            if v > 0.0 {
                cfg.sky_blend_full_at = v;
            }
        }
        if let Some(v) = env_f32("TS_DEPTH_JITTER") {
            if v >= 0.0 {
                cfg.volume_jitter = v;
            }
        }
        if let Some(v) = env_f32("TS_RAMP_DITHER") {
            if v >= 0.0 {
                cfg.ramp_dither = v;
            }
        }
        cfg
    }

    /// Install this config as the ACTIVE depth-contrast config the deep shade dispatch
    /// reads back via [`active`]. An explicit write — no env. The cast applies the live
    /// [`DepthContrastConfig`] resource at its entry; tests apply a constructed one.
    #[inline]
    pub fn apply(&self) {
        ACTIVE_DEPTH.with(|c| c.set(*self));
    }
}

#[inline]
fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
}

#[inline]
fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|raw| {
        let t = raw.trim();
        if t == "1" || t.eq_ignore_ascii_case("true") {
            Some(true)
        } else if t == "0" || t.eq_ignore_ascii_case("false") {
            Some(false)
        } else {
            None
        }
    })
}

// ── active depth-contrast config (explicit per-cast, never env in the hot path) ──
//
// Mirrors `config.rs` / `lod.rs`: the config is read deep inside
// `Material::apply_lighting`, where a plain argument cannot reach without a large API
// churn. Held in a thread-local set ONLY explicitly (by `DepthContrastConfig::apply`,
// from the live resource at the cast boundary, or directly by a test). NOT an env
// global — no `env::var` in the path, so a unit test sets the config directly and
// asserts the resulting shade with zero environment dependence. Stable within a cast
// ⇒ golden-safe; OFF by default ⇒ byte-identical.

thread_local! {
    static ACTIVE_DEPTH: StdCell<DepthContrastConfig> = const {
        StdCell::new(DepthContrastConfig {
            enabled: false,
            near_contrast: 1.0,
            far_contrast: DEPTH_DEFAULT_FAR_CONTRAST,
            fade_to_sky: true,
            sky_blend_full_at: DEPTH_DEFAULT_SKY_BLEND_FULL_AT,
            volume_jitter: DEPTH_DEFAULT_VOLUME_JITTER,
            ramp_dither: DEPTH_DEFAULT_RAMP_DITHER,
        })
    };
}

/// The ACTIVE depth-contrast config for the current cast/thread. Set via
/// [`DepthContrastConfig::apply`]; defaults to [`DepthContrastConfig::default`] (OFF).
/// Read by [`crate::ascii_shader::Material::apply_lighting`].
#[inline]
pub fn active() -> DepthContrastConfig {
    ACTIVE_DEPTH.with(|c| c.get())
}

/// The per-cell CONTRAST factor for a cell at `distance`, interpolated across the LOD
/// `near_end`..`far_start` band: `<= near_end ⇒ near_contrast`, `>= far_start ⇒
/// far_contrast`, linear between. A non-finite distance (sky / miss) ⇒ `far_contrast`
/// (treated as maximally distant). Pure; the hot path inlines this.
#[inline]
pub fn contrast_for(
    distance: f32,
    near_end: f32,
    far_start: f32,
    cfg: &DepthContrastConfig,
) -> f32 {
    if !distance.is_finite() {
        return cfg.far_contrast;
    }
    let d = distance.max(0.0);
    if d <= near_end {
        cfg.near_contrast
    } else if d >= far_start || far_start <= near_end {
        cfg.far_contrast
    } else {
        let t = (d - near_end) / (far_start - near_end);
        cfg.near_contrast + (cfg.far_contrast - cfg.near_contrast) * t
    }
}

/// The `fade_to_sky` blend amount `[0,1]` for a cell at `distance`: `0` at/below
/// `far_start` (no fade), ramping linearly to `1` at `sky_blend_full_at` and beyond.
/// A degenerate band (`full_at <= far_start`) is a hard step at `far_start`. A
/// non-finite distance ⇒ `1` (fully dissolved). Pure.
#[inline]
pub fn sky_blend_for(distance: f32, far_start: f32, full_at: f32) -> f32 {
    if !distance.is_finite() {
        return 1.0;
    }
    let d = distance.max(0.0);
    if d <= far_start {
        0.0
    } else if d >= full_at || full_at <= far_start {
        1.0
    } else {
        (d - far_start) / (full_at - far_start)
    }
}

/// A DETERMINISTIC, hash-STABLE per-cell jitter factor in `[1 - amp, 1 + amp]`, keyed
/// on the cell's WORLD sample coordinate. Because it is a pure function of the quantized
/// world coord (NOT of time or the camera), it is IDENTICAL every frame for a given
/// cell — no flicker, and it survives the cross-frame world-tex cache unchanged.
///
/// The coord is quantized to a fine grid before hashing so two casts that land on
/// essentially the same world texel get the same jitter (otherwise sub-cell camera
/// motion would re-roll the hash and reintroduce shimmer). `amp == 0.0` ⇒ exactly
/// `1.0` (no jitter). A non-finite coord ⇒ `1.0` (safe).
#[inline]
pub fn volume_jitter_factor(coord_x: f32, coord_y: f32, amp: f32) -> f32 {
    if amp <= 0.0 || !coord_x.is_finite() || !coord_y.is_finite() {
        return 1.0;
    }
    // Quantize to 1/16-cell so the hash is stable under sub-cell motion (the cross-frame
    // cache already snaps far texels; this matches that grain).
    const Q: f32 = 16.0;
    let qx = (coord_x * Q).round() as i64;
    let qy = (coord_y * Q).round() as i64;
    // A small integer mix (splitmix64-style finaliser on a 2D-combined key). Pure
    // integer math → deterministic across platforms.
    let mut h = (qx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (qy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    // Map the top bits to [-1, 1].
    let unit = ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    1.0 + unit * amp
}

/// Band width (world cells) below which [`jitter_amplitude_for_band`] starts tapering
/// the volume jitter toward zero. The level `08-material-demo`
/// authors `fog_near = 1 / fog_far = 4` — a band ~2.5 cells wide once floored by
/// [`DEPTH_CONTRAST_MIN_NEAR_END`] — and the flat per-texel jitter, applied over a
/// contrast collapse that completes within that same 2.5 cells, reads as block-colour
/// MOTTLING rather than the intended faint volume grain.
pub const DEPTH_JITTER_SHORT_BAND_MIN: f32 = 2.0;

/// Band width (world cells) at/above which [`jitter_amplitude_for_band`] applies the
/// FULL, un-scaled amplitude — the shipped corpus's ordinary room band (`fog_near..
/// fog_far`) is comfortably wider than this (rooms run 8–25 world cells per the module
/// note above), so an ordinary level's jitter is untouched by this taper.
pub const DEPTH_JITTER_SHORT_BAND_FULL: f32 = 8.0;

/// Scale a base volume-jitter amplitude by the depth-contrast band's width
/// (`far_start - near_end`, world cells): full amplitude at/above
/// [`DEPTH_JITTER_SHORT_BAND_FULL`], smoothly tapering to ZERO at/below
/// [`DEPTH_JITTER_SHORT_BAND_MIN`] (a `smoothstep`, so the taper itself has no hard
/// edge). A short band means the contrast collapse (near→far contrast) completes over
/// only a few world cells, so a flat per-texel jitter amplitude that reads as gentle
/// grain over a normal 8–25-cell room reads as blotchy mottling there — this keeps the
/// EFFECT (contrast collapse, sky-fade) but fades the noisy half out with the band
/// that makes it noisy. Pure; `band_width <= 0` (degenerate) also yields `0.0`.
#[inline]
#[must_use]
pub fn jitter_amplitude_for_band(base_amp: f32, band_width: f32) -> f32 {
    base_amp
        * crate::ascii_shader::smoothstep(
            DEPTH_JITTER_SHORT_BAND_MIN,
            DEPTH_JITTER_SHORT_BAND_FULL,
            band_width,
        )
}

// ── level-authored FOG BAND (owner ask: "far objects less detailed and less
// bright", tying the depth effect to what the level already dims toward) ─────────
//
// Rooms are typically small (8–25 world cells), and the fixed global LOD band
// (near_end=6, far_start=13) either never reaches its far tier inside a small room
// or reaches it too early relative to where the level's own `[atmosphere] fog_near
// / fog_far` (or `[palette] fog_*`) actually starts tinting toward the backdrop.
// Deriving the depth-contrast band from the SAME per-level fog band the distance
// fog pass already uses (`FogParams::near/far`, forwarded to every cast in
// `SampleCtx::fog`) makes the two colour passes agree: as a surface approaches
// `fog_far` it is simultaneously tinting toward the fog colour (the existing fog
// pass) AND losing ink contrast / gaining volume jitter (this pass) AND losing
// glyph variety (the ramp-step collapse below) — one coherent "dissolving into the
// atmosphere" read instead of three cues on three different schedules.

/// Floor on the depth-contrast NEAR boundary (world cells) [`band_for_fog`] derives
/// from `fog_near`. Even a level authoring an aggressive `fog_near` (near 0, e.g. a
/// tight corridor) keeps the immediate foreground — closer than this many cells —
/// at full, untouched contrast, so a wall right beside the player never starts
/// dissolving. `1.5` sits between the engine's unauthored `FOG_NEAR` (1.0) and the
/// smallest authored value in the shipped corpus (`2.0`, `L3`).
pub const DEPTH_CONTRAST_MIN_NEAR_END: f32 = 1.5;

/// Derive the depth-contrast `(near_end, far_start)` band from the level's own
/// authored fog band instead of the fixed global [`crate::lod::LodConfig`]
/// thresholds. `fog_near`/`fog_far` are `ctx.fog.near`/`ctx.fog.far` — ALWAYS
/// populated (either level-authored or the engine's [`crate::ascii_shader::FogParams::DEFAULT`]),
/// so this always returns a usable band; `lod_near_end`/`lod_far_start` are the
/// fallback used only for the degenerate case (non-finite fog, which a parsed
/// level can never produce — only a hand-built test `SampleCtx` could).
///
/// * `near_end` = `fog_near`, floored at [`DEPTH_CONTRAST_MIN_NEAR_END`].
/// * `far_start` = `fog_far` when it is finite and past `near_end`; otherwise the
///   LOD far boundary (kept at least one cell past `near_end`).
///
/// Pure; no allocation.
#[inline]
pub fn band_for_fog(
    fog_near: f32,
    fog_far: f32,
    lod_near_end: f32,
    lod_far_start: f32,
) -> (f32, f32) {
    let near_end = if fog_near.is_finite() {
        fog_near.max(DEPTH_CONTRAST_MIN_NEAR_END)
    } else {
        lod_near_end
    };
    let far_start = if fog_far.is_finite() && fog_far > near_end {
        fog_far
    } else {
        lod_far_start.max(near_end + 1.0)
    };
    (near_end, far_start)
}

/// [`band_for_fog`], additionally widened by a per-material [`crate::lod::LodOverride`]
/// (the SAME "hero object holds detail farther out" lever [`crate::lod::LodConfig`]'s
/// tier resolver already applies) — the PUZZLE-CRITICAL SURFACE exemption (owner ask:
/// "puzzle nodes / node grids must not dim with distance"). Reuses
/// [`crate::lod::LodConfig::effective`] rather than re-deriving the widening rule, so a
/// material authoring `LodOverride::full_detail_to(d)` gets the SAME "stay near/mid
/// past distance `d`" contract for BOTH the glyph-detail tier AND this colour/ramp
/// band — one override, one meaning, instead of a second, glyph-only lever plus a
/// silent colour exception. `ov = None` is byte-identical to [`band_for_fog`] (no
/// override ⇒ `effective` returns the band unchanged).
///
/// `lod_cfg`'s `near_end`/`far_start` are only the FALLBACK [`band_for_fog`] uses for
/// degenerate fog; its `hysteresis`/`far_octaves` are the pass-through `effective`
/// needs to compute a well-formed clamped hysteresis. Pure; no allocation.
#[inline]
pub fn band_for_fog_ov(
    fog_near: f32,
    fog_far: f32,
    lod_cfg: &crate::lod::LodConfig,
    ov: Option<crate::lod::LodOverride>,
) -> (f32, f32) {
    let (near_end, far_start) =
        band_for_fog(fog_near, fog_far, lod_cfg.near_end, lod_cfg.far_start);
    let band_cfg = crate::lod::LodConfig {
        near_end,
        far_start,
        hysteresis: lod_cfg.hysteresis,
        far_octaves: lod_cfg.far_octaves,
    };
    let (eff_near, eff_far, _) = band_cfg.effective(ov);
    (eff_near, eff_far)
}

// ── DETAIL reduction: body-ramp STEP COLLAPSE (owner ask: "less detailed") ────────
//
// The existing depth-ink falloff (`ascii_shader::depth_ink_falloff` /
// `DEPTH_INK_FLOOR`) already thins the body ramp CONTINUOUSLY with distance — but
// that is a brightness cue, not a variety cue: a far wall still has access to every
// one of the body ramp's ~8 distinct inked glyphs, one per world texel, so a
// panning far wall still shows a rich mix of symbols. Collapsing the NUMBER of
// distinct glyphs a tier can resolve to — not just biasing which one — is what
// reads as "this surface stopped carrying fine relief detail" at range. Reuses
// [`crate::lod::LodConfig`]'s tier resolver (and its hysteresis deadband) so the
// glyph-variety step-down happens at an already anti-pop distance boundary instead
// of inventing a second one.

/// Distinct body-ramp brightness LEVELS permitted in the LOD `Far` tier (§ above).
/// `3` = dense / mid / sparse — a coarse silhouette, still legibly inked (the
/// [`crate::ascii_shader`] `RAMP_MIN_INKED` floor is re-applied after collapsing, so
/// a far wall never blanks out from this alone).
pub const DEPTH_FAR_RAMP_STEPS: f32 = 3.0;

/// As [`DEPTH_FAR_RAMP_STEPS`], for the LOD `Mid` tier — a gentler collapse partway
/// between the near tier's full ~8-step ramp and the far tier's 3-step ramp.
pub const DEPTH_MID_RAMP_STEPS: f32 = 5.0;

/// Quantize a `[0,1]` ramp-brightness value onto `steps` discrete, evenly-spaced
/// levels (`0`, `1/(steps-1)`, …, `1`). `steps == None` (or `<= 1.0`) is the
/// IDENTITY — no collapse, so the NEAR tier (which resolves `None` from
/// [`ramp_steps_for`]) reproduces the un-collapsed ramp exactly. Pure.
#[inline]
#[must_use]
pub fn quantize_ramp_steps(v: f32, steps: Option<f32>) -> f32 {
    match steps {
        Some(steps) if steps > 1.0 => {
            let v = v.clamp(0.0, 1.0);
            (v * (steps - 1.0)).round() / (steps - 1.0)
        }
        _ => v,
    }
}

/// The ramp-step collapse for a cell at `distance` (world cells): resolves the
/// [`crate::lod::LodTier`] for a band derived from the level's fog (`band_for_fog_ov`,
/// using `lod_cfg`'s own hysteresis/octave fields as the pass-through and honouring
/// the cell's material `ov` — the same per-material [`crate::lod::LodOverride`] the
/// glyph-detail LOD tier already reads off `SampleCtx::lod_override`, so a
/// puzzle-critical material authoring `full_detail_to` is exempt from the DETAIL half
/// of the effect too, not just the colour half applied in
/// [`crate::ascii_shader::Material::apply_lighting`]'s 4b block), then maps the tier
/// to the step count [`quantize_ramp_steps`] should collapse onto — `None` for `Near`
/// (no collapse), [`DEPTH_MID_RAMP_STEPS`] / [`DEPTH_FAR_RAMP_STEPS`] for `Mid`/`Far`.
/// Callers gate this on [`DepthContrastConfig::enabled`] (see
/// [`crate::ascii_shader::Material::apply_lighting`]) so `TS_DEPTH=0` restores the
/// un-collapsed ramp exactly, matching the colour half of the effect. Pure.
#[inline]
pub fn ramp_steps_for(
    distance: f32,
    fog_near: f32,
    fog_far: f32,
    lod_cfg: &crate::lod::LodConfig,
    ov: Option<crate::lod::LodOverride>,
) -> Option<f32> {
    let (near_end, far_start) = band_for_fog_ov(fog_near, fog_far, lod_cfg, ov);
    let band_cfg = crate::lod::LodConfig {
        near_end,
        far_start,
        hysteresis: lod_cfg.hysteresis,
        far_octaves: lod_cfg.far_octaves,
    };
    match band_cfg.tier_for(distance, None) {
        crate::lod::LodTier::Near => None,
        crate::lod::LodTier::Mid => Some(DEPTH_MID_RAMP_STEPS),
        crate::lod::LodTier::Far => Some(DEPTH_FAR_RAMP_STEPS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_default_stays_off_and_byte_identical_safe() {
        // `DepthContrastConfig::default()` (and the thread-local's initial value)
        // MUST stay OFF: it is the baseline every colour-exact unit test elsewhere
        // in this crate is written against. The SHIPPED default lives in
        // `from_env()` only (see `from_env_defaults_enabled_unless_ts_depth_says_off`
        // below) — this is a deliberate split, not an oversight.
        let c = DepthContrastConfig::default();
        assert!(
            !c.enabled,
            "struct Default must stay OFF ⇒ byte-identical to today"
        );
        assert_eq!(c.near_contrast, 1.0);
        assert_eq!(c.far_contrast, DEPTH_DEFAULT_FAR_CONTRAST);
        assert!(c.fade_to_sky);
        assert_eq!(c.sky_blend_full_at, DEPTH_DEFAULT_SKY_BLEND_FULL_AT);
        assert_eq!(c.volume_jitter, DEPTH_DEFAULT_VOLUME_JITTER);
    }

    /// THE OFF-SWITCH CONTRACT: `from_env()` with no `TS_DEPTH*` vars set ships
    /// `enabled = true` (the distance-depth effect is ON by default), and every
    /// OTHER field matches the struct `Default` (so `TS_DEPTH=0` — simulated here by
    /// constructing the config directly, since mutating process env is not test-safe
    /// under a parallel runner — reproduces the historical renderer exactly).
    #[test]
    fn from_env_defaults_enabled_unless_ts_depth_says_off() {
        // No TS_DEPTH* set in this test process.
        let shipped = DepthContrastConfig::from_env();
        assert!(
            shipped.enabled,
            "shipped default (from_env, no TS_DEPTH) must be ON"
        );
        let off = DepthContrastConfig {
            enabled: false,
            ..shipped
        };
        assert_eq!(
            off,
            DepthContrastConfig::default(),
            "TS_DEPTH=0 (enabled=false, every other field default) must equal the \
             struct Default ⇒ byte-identical to the pre-feature renderer"
        );
    }

    #[test]
    fn apply_threads_the_active_config_without_env() {
        DepthContrastConfig {
            enabled: true,
            near_contrast: 0.9,
            far_contrast: 0.2,
            fade_to_sky: false,
            sky_blend_full_at: 28.0,
            volume_jitter: 0.1,
            ramp_dither: 0.0,
        }
        .apply();
        let a = active();
        assert!(a.enabled);
        assert_eq!(a.far_contrast, 0.2);
        assert!(!a.fade_to_sky);
        // Restore default for sibling tests on this thread.
        DepthContrastConfig::default().apply();
        assert_eq!(active(), DepthContrastConfig::default());
    }

    #[test]
    fn contrast_interpolates_across_the_lod_band() {
        let c = DepthContrastConfig::default(); // near 1.0, far 0.15
        let (near_end, far_start) = (6.0, 13.0);
        assert_eq!(
            contrast_for(1.0, near_end, far_start, &c),
            1.0,
            "near = full"
        );
        assert_eq!(
            contrast_for(6.0, near_end, far_start, &c),
            1.0,
            "at near_end = full"
        );
        assert_eq!(
            contrast_for(20.0, near_end, far_start, &c),
            0.15,
            "far = collapsed"
        );
        let mid = contrast_for(9.5, near_end, far_start, &c); // halfway
        assert!(
            (mid - 0.575).abs() < 1e-5,
            "midpoint interpolates: got {mid}"
        );
    }

    #[test]
    fn contrast_non_finite_is_far() {
        let c = DepthContrastConfig::default();
        assert_eq!(contrast_for(f32::INFINITY, 6.0, 13.0, &c), c.far_contrast);
        assert_eq!(contrast_for(f32::NAN, 6.0, 13.0, &c), c.far_contrast);
    }

    #[test]
    fn sky_blend_ramps_beyond_far_start() {
        assert_eq!(
            sky_blend_for(10.0, 13.0, 32.0),
            0.0,
            "below far_start = no fade"
        );
        assert_eq!(
            sky_blend_for(13.0, 13.0, 32.0),
            0.0,
            "at far_start = no fade"
        );
        assert_eq!(
            sky_blend_for(32.0, 13.0, 32.0),
            1.0,
            "at full_at = full fade"
        );
        assert_eq!(
            sky_blend_for(99.0, 13.0, 32.0),
            1.0,
            "beyond = clamped full"
        );
        let mid = sky_blend_for(22.5, 13.0, 32.0); // halfway 13..32
        assert!((mid - 0.5).abs() < 1e-5, "midpoint: got {mid}");
    }

    #[test]
    fn volume_jitter_is_hash_stable_across_calls() {
        // The SAME world coord must give the SAME jitter every call (no flicker, cache-safe).
        let a = volume_jitter_factor(3.25, 7.5, 0.06);
        let b = volume_jitter_factor(3.25, 7.5, 0.06);
        assert_eq!(
            a, b,
            "identical coord ⇒ identical jitter (deterministic, no flicker)"
        );
        // Within the ± amplitude band.
        assert!(
            (a - 1.0).abs() <= 0.06 + 1e-6,
            "jitter stays within ±amp: {a}"
        );
        // A different coord generally rolls a different factor.
        let other = volume_jitter_factor(11.7, -0.3, 0.06);
        assert_ne!(a, other, "distinct coords decorrelate");
    }

    #[test]
    fn volume_jitter_zero_amp_is_identity() {
        assert_eq!(volume_jitter_factor(1.0, 2.0, 0.0), 1.0);
        assert_eq!(
            volume_jitter_factor(f32::NAN, 2.0, 0.06),
            1.0,
            "non-finite is safe"
        );
    }

    // ── jitter amplitude scales down with a short fog band (08-material-demo mottling) ──

    #[test]
    fn jitter_amplitude_for_band_is_full_at_an_ordinary_room_width() {
        // A shipped room's fog band (e.g. L1's 3..16 ⇒ width 13) is well past the
        // "full amplitude" floor — untouched, byte-identical to the pre-fix flat amp.
        let amp = jitter_amplitude_for_band(0.06, 13.0);
        assert!(
            (amp - 0.06).abs() < 1e-6,
            "ordinary room width ⇒ full amplitude: got {amp}"
        );
    }

    #[test]
    fn jitter_amplitude_for_band_vanishes_for_08_material_demos_short_band() {
        // 08-material-demo: fog_near=1/fog_far=4 ⇒ band_for_fog width ≈ 2.5 (near_end
        // floored to DEPTH_CONTRAST_MIN_NEAR_END=1.5). Must read as materially reduced
        // — the mottling this fixes — not the flat 0.06 the pre-fix code always used.
        let (near_end, far_start) = band_for_fog(1.0, 4.0, 6.0, 13.0);
        let width = far_start - near_end;
        let amp = jitter_amplitude_for_band(0.06, width);
        assert!(
            amp < 0.06 * 0.3,
            "short band must sharply reduce amplitude: width={width} amp={amp}"
        );
    }

    #[test]
    fn jitter_amplitude_for_band_is_zero_at_or_below_the_min_width() {
        assert_eq!(
            jitter_amplitude_for_band(0.06, DEPTH_JITTER_SHORT_BAND_MIN),
            0.0
        );
        assert_eq!(
            jitter_amplitude_for_band(0.06, 0.5),
            0.0,
            "narrower than the min ⇒ still zero"
        );
        assert_eq!(
            jitter_amplitude_for_band(0.06, -1.0),
            0.0,
            "degenerate width ⇒ zero, never negative"
        );
    }

    #[test]
    fn jitter_amplitude_for_band_is_monotonic_nondecreasing() {
        let widths = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0];
        let mut prev = 0.0f32;
        for w in widths {
            let amp = jitter_amplitude_for_band(0.06, w);
            assert!(
                amp >= prev - 1e-6,
                "amplitude must not decrease as the band widens: w={w} amp={amp} prev={prev}"
            );
            prev = amp;
        }
    }

    // ── level-authored fog band ───────────────────────────────────────────────

    #[test]
    fn band_for_fog_uses_the_authored_fog_band() {
        // L1: fog_near=3, fog_far=16.
        assert_eq!(band_for_fog(3.0, 16.0, 6.0, 13.0), (3.0, 16.0));
        // L3: fog_near=2, fog_far=14.
        assert_eq!(band_for_fog(2.0, 14.0, 6.0, 13.0), (2.0, 14.0));
    }

    #[test]
    fn band_for_fog_floors_a_very_close_fog_near() {
        // An aggressive fog_near (below the floor) must not start the collapse at
        // the player's face.
        let (near_end, _) = band_for_fog(0.2, 20.0, 6.0, 13.0);
        assert_eq!(near_end, DEPTH_CONTRAST_MIN_NEAR_END);
    }

    #[test]
    fn band_for_fog_falls_back_to_lod_on_degenerate_fog() {
        // Non-finite fog (only a hand-built test SampleCtx can produce this — a
        // parsed level never does) falls back to the LOD thresholds.
        assert_eq!(band_for_fog(f32::NAN, f32::NAN, 6.0, 13.0), (6.0, 13.0));
        // fog_far <= near_end (degenerate authoring) also falls back for far_start.
        let (near_end, far_start) = band_for_fog(10.0, 5.0, 6.0, 13.0);
        assert_eq!(near_end, 10.0);
        assert!(
            far_start > near_end,
            "far_start must stay past near_end: {far_start}"
        );
    }

    #[test]
    fn band_for_fog_matches_the_unauthored_engine_default() {
        // FogParams::DEFAULT is (1.0, 26.0) — an unauthored level still gets a
        // sensible, floored band, not a panic or an inverted one.
        let (near_end, far_start) = band_for_fog(1.0, 26.0, 6.0, 13.0);
        assert_eq!(near_end, DEPTH_CONTRAST_MIN_NEAR_END);
        assert_eq!(far_start, 26.0);
    }

    // ── ramp-step collapse (the DETAIL half of the effect) ────────────────────

    #[test]
    fn quantize_ramp_steps_none_is_identity() {
        for v in [0.0, 0.22, 0.5, 0.777, 1.0] {
            assert_eq!(
                quantize_ramp_steps(v, None),
                v,
                "no collapse ⇒ identity at v={v}"
            );
        }
    }

    #[test]
    fn quantize_ramp_steps_snaps_to_discrete_levels() {
        // 3 steps ⇒ levels {0.0, 0.5, 1.0}.
        assert_eq!(quantize_ramp_steps(0.0, Some(3.0)), 0.0);
        assert_eq!(
            quantize_ramp_steps(0.4, Some(3.0)),
            0.5,
            "0.4 rounds to the 0.5 level"
        );
        assert_eq!(
            quantize_ramp_steps(0.6, Some(3.0)),
            0.5,
            "0.6 also rounds to 0.5"
        );
        assert_eq!(quantize_ramp_steps(0.9, Some(3.0)), 1.0);
        assert_eq!(quantize_ramp_steps(1.0, Some(3.0)), 1.0);
        // Fewer than 4 distinct outputs total across the whole [0,1] domain.
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            seen.insert(quantize_ramp_steps(v, Some(3.0)).to_bits());
        }
        assert!(
            seen.len() <= 3,
            "3-step collapse must yield at most 3 distinct outputs, got {}",
            seen.len()
        );
    }

    #[test]
    fn quantize_ramp_steps_degenerate_steps_is_identity() {
        assert_eq!(
            quantize_ramp_steps(0.42, Some(1.0)),
            0.42,
            "steps<=1 is a no-op"
        );
        assert_eq!(quantize_ramp_steps(0.42, Some(0.0)), 0.42);
    }

    #[test]
    fn ramp_steps_for_is_none_in_the_near_tier() {
        let lod = crate::lod::LodConfig::default();
        // Well inside L1's authored fog band (3..16), close to the player.
        assert_eq!(
            ramp_steps_for(1.0, 3.0, 16.0, &lod, None),
            None,
            "near cell: no collapse"
        );
    }

    #[test]
    fn ramp_steps_for_collapses_more_aggressively_with_distance() {
        let lod = crate::lod::LodConfig::default();
        // Fog band (3, 16) ⇒ near_end=3, far_start=16, hysteresis 0.75 (default).
        let mid = ramp_steps_for(9.5, 3.0, 16.0, &lod, None); // well inside the mid tier
        let far = ramp_steps_for(20.0, 3.0, 16.0, &lod, None); // past far_start
        assert_eq!(mid, Some(DEPTH_MID_RAMP_STEPS));
        assert_eq!(far, Some(DEPTH_FAR_RAMP_STEPS));
    }

    // Compile-time: far must collapse onto FEWER steps than mid (a constant
    // condition, checked by the compiler instead of a runtime `assert!`).
    const _: () = assert!(DEPTH_FAR_RAMP_STEPS < DEPTH_MID_RAMP_STEPS);

    #[test]
    fn ramp_steps_for_is_hysteresis_stable_at_the_boundary() {
        // The tier resolver's static deadband (§ LodConfig) means a distance a
        // WHISKER inside the near_end boundary still resolves Near, not a
        // hair-trigger Mid — proving the collapse rides the same anti-pop
        // machinery as every other LOD-driven decision, not a fresh hard cutoff.
        let lod = crate::lod::LodConfig::default();
        let (near_end, _) = band_for_fog(3.0, 16.0, lod.near_end, lod.far_start);
        assert_eq!(
            ramp_steps_for(near_end + 0.1, 3.0, 16.0, &lod, None),
            None,
            "inside the deadband holds Near"
        );
        assert_eq!(
            ramp_steps_for(near_end + 10.0, 3.0, 16.0, &lod, None),
            Some(DEPTH_MID_RAMP_STEPS),
            "well past the deadband steps to Mid"
        );
    }

    // ── per-material LodOverride exemption (owner ask: puzzle-critical surfaces —
    // nodes / node grids — must not lose contrast or glyph variety with distance) ──

    #[test]
    fn ramp_steps_for_a_full_detail_override_holds_near_past_the_global_far_tier() {
        let lod = crate::lod::LodConfig::default();
        // Fog band (3, 16): at distance 20 the DEFAULT material is well past
        // far_start (Far tier, collapsed) — but a material authoring
        // `full_detail_to(40)` must stay Near (no collapse) at the SAME distance.
        let default_mat = ramp_steps_for(20.0, 3.0, 16.0, &lod, None);
        let hero = Some(crate::lod::LodOverride::full_detail_to(40.0));
        let hero_mat = ramp_steps_for(20.0, 3.0, 16.0, &lod, hero);
        assert_eq!(
            default_mat,
            Some(DEPTH_FAR_RAMP_STEPS),
            "default material collapses far"
        );
        assert_eq!(
            hero_mat, None,
            "hero override stays full detail (no collapse) at the same distance"
        );
    }

    #[test]
    fn band_for_fog_ov_none_matches_band_for_fog() {
        let lod = crate::lod::LodConfig::default();
        let plain = band_for_fog(3.0, 16.0, lod.near_end, lod.far_start);
        let ov_none = band_for_fog_ov(3.0, 16.0, &lod, None);
        assert_eq!(
            plain, ov_none,
            "no override ⇒ byte-identical to band_for_fog"
        );
    }

    #[test]
    fn band_for_fog_ov_widens_the_band_for_a_hero_material() {
        let lod = crate::lod::LodConfig::default();
        let (near_end, far_start) = band_for_fog(3.0, 16.0, lod.near_end, lod.far_start);
        let ov = Some(crate::lod::LodOverride::full_detail_to(50.0));
        let (ov_near, ov_far) = band_for_fog_ov(3.0, 16.0, &lod, ov);
        assert!(
            ov_near >= 50.0,
            "near_end pushed out to at least the override distance: {ov_near}"
        );
        assert!(
            ov_far > far_start,
            "far_start widens too: {ov_far} vs plain {far_start}"
        );
    }
}
