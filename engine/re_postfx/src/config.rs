//! The configurable effect set + the [`PostfxConfig`] aggregate and its presets.
//!
//! Each effect is a small `*Fx` struct: an `enabled` flag plus its own params.
//! [`PostfxConfig`] holds one of each plus the master [`PostfxConfig::intensity`]
//! (0..1) that scales every effect where scaling is meaningful — one mood knob.
//!
//! Effects are deliberately COMBINABLE: enable any subset, and [`apply_postfx`]
//! composes them in a fixed pipeline order. Presets ([`PostfxConfig::off`],
//! `clean`, `subtle`, `cinematic`, `toon`, `crt`, `dream`) bundle tasteful
//! combinations.
//!
//! # The ordered stage abstraction (extensibility)
//!
//! The pipeline order is a CONTRACT, not an accident (§15 strict stack: atmosphere
//! → light → structure → distortion → colour → stylization → quantization). It is
//! enumerated ONCE in [`PostfxStage`] and its [`PostfxStage::ORDER`] slice; the
//! driver in [`crate::apply_postfx_ex`] walks exactly that list. To add a new
//! effect you:
//!   1. add its `*Fx` config field here (default OFF so the identity preset stays
//!      byte-identical),
//!   2. add a [`PostfxStage`] variant at its correct position in `ORDER`,
//!   3. add its `effects::<name>` module and one match arm in the driver.
//!
//! No unrelated stage is touched, and the **dither-last invariant** — dither is the
//! final stage of `ORDER` and may be DEFERRED past an external glitch layer via
//! `skip_dither` — is preserved structurally.
//!
//! [`apply_postfx`]: crate::apply_postfx

use bevy::prelude::Resource;

/// The fixed, ordered list of post-process stages — the single source of truth for
/// pipeline order (spec §15 «строгий порядок»). The driver
/// ([`crate::apply_postfx_ex`]) walks [`PostfxStage::ORDER`] top-to-bottom; adding
/// a stage is "insert a variant at the right index", never a reorder of existing
/// ones.
///
/// Order logic (spec §15): atmosphere/occlusion → light → structure → distortion →
/// colour grade/tint/quantize → vignette → CRT stylization (scanlines, grain) →
/// **DITHER LAST**. The frozen subsequence `Ssao → Bloom → Edge → Dof → Grade →
/// Vignette → Dither` is preserved; new stages slot strictly BETWEEN those, never
/// reordering them, and `Dither` is always last (and uniquely deferrable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostfxStage {
    /// Contact-shadow / ambient occlusion from depth (darkens recessed bg).
    Ssao,
    /// Light bleed from bright cells (must run before structure — §15).
    Bloom,
    /// Toon-ink silhouettes on depth/surface/luma discontinuities (after light).
    Edge,
    /// Depth-of-field distance blur + desaturation.
    Dof,
    /// Chromatic aberration: fg/bg channel split at edges ("distortion" slot).
    ChromaticAberration,
    /// Filmic tonemap + contrast/saturation/lift-gamma-gain/hue (the mood grade).
    Grade,
    /// White-balance: colour-temperature warm/cool + magenta/green tint.
    ColorTemp,
    /// Posterize / colour-quantize for a flat ASCII-poster look.
    Posterize,
    /// Radial edge darkening.
    Vignette,
    /// CRT scanline / shadow-mask darkening.
    Scanlines,
    /// World/cell-anchored film grain (stable per time bucket, no per-frame shimmer).
    FilmGrain,
    /// Ordered-Bayer dither + palette quantization — ALWAYS LAST; uniquely
    /// deferrable past an external glitch layer (`skip_dither`).
    Dither,
}

impl PostfxStage {
    /// The canonical pipeline order. The driver iterates this exactly; `Dither` is
    /// the terminal element by contract.
    pub const ORDER: [PostfxStage; 12] = [
        PostfxStage::Ssao,
        PostfxStage::Bloom,
        PostfxStage::Edge,
        PostfxStage::Dof,
        PostfxStage::ChromaticAberration,
        PostfxStage::Grade,
        PostfxStage::ColorTemp,
        PostfxStage::Posterize,
        PostfxStage::Vignette,
        PostfxStage::Scanlines,
        PostfxStage::FilmGrain,
        PostfxStage::Dither,
    ];
}

// ───────────────────────────── per-effect configs ───────────────────────────

/// **Edge-outline (toon ink).** Detects a depth OR surface-kind discontinuity
/// versus the 4-neighbours and darkens the fg (ink stamp) on an edge so geometry
/// reads outlined — a cel-shaded silhouette pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Relative depth jump (fraction of the local depth) that counts as an edge.
    /// Lower = more edges. A surface-kind change is always an edge regardless.
    pub depth_threshold: f32,
    /// How hard the ink darkens the fg on an edge (0 = none, 1 = black). Scaled
    /// by the master intensity.
    pub strength: f32,
    /// Luma-discontinuity threshold (0..1): a fg-brightness jump versus a
    /// 4-neighbour beyond this ALSO marks an edge, so a flat wall painted with a
    /// bright texture line outlines too — not just geometry seams. `0` disables the
    /// luma test (pure depth/surface edges, the original behaviour).
    pub luma_threshold: f32,
}

/// **SSAO / contact shadow.** Screen-space ambient occlusion from the depth
/// buffer: a cell whose depth is notably GREATER than its neighbours' (a recess,
/// or the far side of an edge) gets its bg darkened proportionally.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsaoFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Neighbour sampling radius in cells (1 or 2). Larger = softer, wider AO.
    pub radius: u32,
    /// Maximum bg darkening at full occlusion (0..1). Scaled by master intensity.
    pub strength: f32,
    /// Depth difference (world units) below which no occlusion is applied — kills
    /// AO noise on flat surfaces. Larger = only deep recesses darken.
    pub bias: f32,
}

/// **Bloom / glow.** Thresholds bright cells (fg luma above `threshold`) and
/// bleeds their brightness into neighbours within `radius` (a small box blur of
/// the bright mask, added back to neighbour fg + bg) — a soft glow halo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Fg-luma cutoff above which a cell is a bloom source (0..1).
    pub threshold: f32,
    /// Soft-knee width below the threshold (0..1). A cell whose luma sits within
    /// `knee` BELOW `threshold` contributes partially (smooth ramp) instead of a
    /// hard cutoff — kills the binary "pops in / pops out" shimmer of bright cells
    /// crossing the threshold. `0` = hard knee (the original behaviour).
    pub knee: f32,
    /// Bleed radius in cells (1 or 2).
    pub radius: u32,
    /// How strongly the bleed adds to neighbours (0..1+). Scaled by master.
    pub intensity: f32,
}

/// **DoF / distance blur.** Cells beyond `focus_dist` soften: their color blends
/// toward the 4-neighbour average and desaturates, ramping over `range` up to
/// `max_blur`. Cheap depth-driven atmosphere / focus.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Depth (world units) at/under which a cell is in perfect focus.
    pub focus_dist: f32,
    /// Depth span over which blur ramps from 0 to `max_blur`.
    pub range: f32,
    /// Maximum blur amount (0..1): blend-to-neighbour-average fraction at the far
    /// end. Also drives the desaturation. Scaled by master intensity.
    pub max_blur: f32,
}

/// **Color-grade / tonemap.** A filmic tonemap plus contrast, saturation and
/// lift/gamma/gain, with an optional hue rotation — the mood grade. Applied to
/// every cell's fg AND bg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradeFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Apply a filmic tonemap (rolls off highlights). The curve is selected by
    /// [`aces`](Self::aces): `false` = the cheap Reinhard-ish `2x/(1+x)`,
    /// `true` = the Narkowicz ACES approximation (richer shoulder/toe, the default
    /// filmic look).
    pub tonemap: bool,
    /// When `tonemap` is on, use the ACES filmic approximation instead of the
    /// Reinhard-ish curve. Punchier contrast and a cinematic highlight rolloff.
    pub aces: bool,
    /// Contrast multiplier around mid-grey (1 = none, >1 punchier).
    pub contrast: f32,
    /// Saturation multiplier (1 = none, 0 = greyscale, >1 vivid).
    pub saturation: f32,
    /// Lift (added to shadows), gamma (power), gain (multiplied) — the classic
    /// three-way grade, applied per channel after tonemap.
    pub lift: f32,
    /// Gamma (power curve) — the middle of the lift/gamma/gain three-way grade.
    pub gamma: f32,
    /// Gain (multiplied) — the highlights term of the lift/gamma/gain grade.
    pub gain: f32,
    /// Hue rotation in degrees (0 = none). A cheap chroma rotate for a tint.
    pub hue_shift: f32,
}

/// **Vignette.** Radial darkening toward the frame edges — a soft oval falloff
/// multiplied onto fg + bg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VignetteFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// How dark the corners go at full strength (0 = none, 1 = black corners).
    /// Scaled by master intensity.
    pub amount: f32,
    /// Falloff softness: the normalized radius at which darkening begins (0..1).
    /// Larger = darkening confined to the very edge.
    pub softness: f32,
}

/// **Dither (ordered Bayer 4×4).** Adds a deterministic, per-cell ordered
/// dithering pattern to fg/bg to smooth banding (helps in 256-color mode) or as a
/// stylistic texture. Spatially fixed (no time term) so it is golden-stable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitherFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Dither amplitude in 0..1 color units (the peak ± wobble). Scaled by master.
    pub amount: f32,
}

/// **Chromatic aberration.** A lens-fringe split: near frame edges (and across
/// luma discontinuities) the red and blue channels are sampled from offset
/// neighbours, leaving a cyan/red ghost. Terminal-appropriate — there are no real
/// subpixels, so we offset whole-cell colour CHANNELS toward neighbour cells
/// rather than sub-pixel shifting. Strongest at the frame edge (radial), like a
/// real lens. Reads the clean `src`; writes fg + bg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromaticFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Peak channel offset in CELLS at the frame corner (0..3). Scaled by master.
    pub amount: f32,
    /// Radius (0..1) inside which there is no fringe; it ramps from here to the
    /// corner. Larger = aberration confined to the very edge.
    pub edge_only: f32,
}

/// **Colour temperature / tint (white balance).** A per-channel multiply that
/// warms (amber) or cools (blue) the whole frame, plus an orthogonal magenta↔green
/// tint axis. The cheap mood/atmosphere knob that does NOT touch contrast or
/// saturation (use [`GradeFx`] for those). Applied to fg + bg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorTempFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Warm↔cool, −1 (cold blue) .. 0 (neutral) .. +1 (warm amber). Scaled by master.
    pub temperature: f32,
    /// Magenta↔green, −1 (magenta) .. 0 .. +1 (green). Scaled by master.
    pub tint: f32,
}

/// **Posterize / colour-quantize.** Snaps each channel to `levels` evenly-spaced
/// steps for a flat, banded poster / ASCII-art look. Distinct from dither (which
/// quantizes to the terminal PALETTE last); posterize is a deliberate stylistic
/// banding applied in the float domain before vignette.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosterizeFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Steps per channel (2..=64). 2 = harsh, ~8 = gentle. Fewer = more banded.
    pub levels: u32,
}

/// **Scanlines / CRT shadow-mask.** Darkens alternate rows (scanline) and,
/// optionally, applies a 3-cell-period column mask (aperture-grille feel). A
/// spatial, time-free effect — golden-stable. Multiplies fg + bg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanlineFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// How dark the off-scanline rows go (0 = none, 1 = black). Scaled by master.
    pub strength: f32,
    /// Row period in cells (1 = every other row darkened, 2 = every 3rd, …).
    pub period: u32,
    /// Aperture-grille column mask strength (0 = pure scanlines). Scaled by master.
    pub mask: f32,
}

/// **Film grain.** A subtle luminance wobble keyed on `(cell, time_bucket)`, NOT
/// per-frame white noise — so a STATIC cell at a FIXED time bucket is byte-stable
/// (no shimmer). The caller advances [`time_bucket`](Self::time_bucket) from a SLOW
/// clock (e.g. a few buckets/sec) so grain crawls gently instead of strobing; the
/// plugin derives it from `Time`. Because `apply_postfx` is pure in `(fb, cfg)`,
/// the bucket lives in the config and the determinism contract is preserved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmGrainFx {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Grain amplitude in 0..1 luma units (peak ± wobble). Scaled by master.
    pub amount: f32,
    /// Slow-time bucket — grain is anchored on `(col, row, time_bucket)`. Hold it
    /// constant ⇒ frozen grain (golden-stable); bump it ⇒ the grain re-rolls. The
    /// plugin advances it from a slow clock so it never shimmers per render-frame.
    pub time_bucket: u32,
}

// ───────────────────────────────── aggregate ────────────────────────────────

/// The full post-process configuration: one of every effect plus the master
/// [`intensity`](Self::intensity) (0..1) that scales the effects where scaling is
/// meaningful. Exposed as a Bevy [`Resource`] so the game can set it per-level or
/// from a settings toggle.
///
/// Default is [`PostfxConfig::off`] — the identity — so shipped levels render
/// unchanged unless something opts in.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct PostfxConfig {
    /// Master mood scalar 0..1. `0` ⇒ a guaranteed identity even if individual
    /// effects are flagged on (the off fast-path). `1` ⇒ each effect at its
    /// authored strength. Effects that take a strength/amount multiply it by this.
    pub intensity: f32,
    /// Toon-ink outline pass (see [`EdgeFx`]).
    pub edge: EdgeFx,
    /// Contact-shadow / ambient-occlusion pass (see [`SsaoFx`]).
    pub ssao: SsaoFx,
    /// Bloom / glow pass (see [`BloomFx`]).
    pub bloom: BloomFx,
    /// Depth-of-field blur pass (see [`DofFx`]).
    pub dof: DofFx,
    /// Chromatic-aberration pass (see [`ChromaticFx`]).
    pub chromatic: ChromaticFx,
    /// Colour-grade / tonemap pass (see [`GradeFx`]).
    pub grade: GradeFx,
    /// White-balance (temperature/tint) pass (see [`ColorTempFx`]).
    pub color_temp: ColorTempFx,
    /// Posterize / colour-quantize pass (see [`PosterizeFx`]).
    pub posterize: PosterizeFx,
    /// Radial edge-darkening pass (see [`VignetteFx`]).
    pub vignette: VignetteFx,
    /// CRT scanline / shadow-mask pass (see [`ScanlineFx`]).
    pub scanlines: ScanlineFx,
    /// Film-grain pass (see [`FilmGrainFx`]).
    pub film_grain: FilmGrainFx,
    /// Ordered-Bayer dither pass, always last (see [`DitherFx`]).
    pub dither: DitherFx,
}

impl Default for PostfxConfig {
    /// The default is OFF — the additive guarantee for shipped levels.
    fn default() -> Self {
        Self::off()
    }
}

impl PostfxConfig {
    /// Every effect disabled, master intensity 0. [`apply_postfx`] is the identity
    /// on this config (the frame is left byte-identical).
    ///
    /// [`apply_postfx`]: crate::apply_postfx
    pub fn off() -> Self {
        PostfxConfig {
            intensity: 0.0,
            edge: EdgeFx {
                enabled: false,
                depth_threshold: 0.15,
                strength: 0.6,
                luma_threshold: 0.0,
            },
            ssao: SsaoFx {
                enabled: false,
                radius: 1,
                strength: 0.6,
                bias: 0.25,
            },
            bloom: BloomFx {
                enabled: false,
                threshold: 0.7,
                knee: 0.0,
                radius: 1,
                intensity: 0.6,
            },
            dof: DofFx {
                enabled: false,
                focus_dist: 4.0,
                range: 8.0,
                max_blur: 0.7,
            },
            chromatic: ChromaticFx {
                enabled: false,
                amount: 1.0,
                edge_only: 0.5,
            },
            grade: GradeFx {
                enabled: false,
                tonemap: true,
                aces: true,
                contrast: 1.0,
                saturation: 1.0,
                lift: 0.0,
                gamma: 1.0,
                gain: 1.0,
                hue_shift: 0.0,
            },
            color_temp: ColorTempFx {
                enabled: false,
                temperature: 0.0,
                tint: 0.0,
            },
            posterize: PosterizeFx {
                enabled: false,
                levels: 8,
            },
            vignette: VignetteFx {
                enabled: false,
                amount: 0.5,
                softness: 0.5,
            },
            scanlines: ScanlineFx {
                enabled: false,
                strength: 0.3,
                period: 1,
                mask: 0.0,
            },
            film_grain: FilmGrainFx {
                enabled: false,
                amount: 0.05,
                time_bucket: 0,
            },
            dither: DitherFx {
                enabled: false,
                amount: 0.04,
            },
        }
    }

    /// A neutral-but-finished look: a touch of contact AO + the filmic tonemap and
    /// a whisper of grain to break up flat banding, with NO colour push. The
    /// "looks rendered, not tinted" preset. Master intensity 0.5.
    pub fn clean() -> Self {
        let mut c = Self::off();
        c.intensity = 0.5;
        c.ssao.enabled = true;
        c.ssao.strength = 0.35;
        c.grade.enabled = true;
        c.grade.tonemap = true;
        c.grade.aces = true;
        c.grade.contrast = 1.05;
        c.film_grain.enabled = true;
        c.film_grain.amount = 0.03;
        c
    }

    /// A light, tasteful preset: a soft vignette + a gentle filmic grade + a
    /// whisper of SSAO. Reads as "graded but natural". Master intensity 0.4.
    pub fn subtle() -> Self {
        let mut c = Self::off();
        c.intensity = 0.4;
        c.ssao.enabled = true;
        c.ssao.strength = 0.4;
        c.grade.enabled = true;
        c.grade.contrast = 1.08;
        c.grade.saturation = 1.05;
        c.vignette.enabled = true;
        c.vignette.amount = 0.35;
        c
    }

    /// The marquee filmic preset: color-grade + bloom + vignette + depth-of-field
    /// (+ contact AO) for a moody, cinematic depth read. Master intensity 0.8.
    pub fn cinematic() -> Self {
        let mut c = Self::off();
        c.intensity = 0.8;
        c.ssao.enabled = true;
        c.ssao.strength = 0.6;
        c.bloom.enabled = true;
        c.bloom.threshold = 0.6;
        c.bloom.knee = 0.2; // soft knee: bright cells fade in, no threshold pop
        c.bloom.radius = 2;
        c.bloom.intensity = 0.7;
        c.dof.enabled = true;
        c.dof.focus_dist = 3.5;
        c.dof.range = 9.0;
        c.dof.max_blur = 0.7;
        c.grade.enabled = true;
        c.grade.tonemap = true;
        c.grade.aces = true; // ACES filmic shoulder for the cinematic rolloff
        c.grade.contrast = 1.15;
        c.grade.saturation = 1.1;
        c.grade.lift = 0.02;
        c.grade.gain = 1.05;
        c.color_temp.enabled = true;
        c.color_temp.temperature = -0.18; // a cool teal lean (white balance, not hue)
        c.vignette.enabled = true;
        c.vignette.amount = 0.55;
        c.vignette.softness = 0.45;
        c.film_grain.enabled = true;
        c.film_grain.amount = 0.035;
        c
    }

    /// A cel-shaded look: hard edge-outline ink + a flat, punchy grade (high
    /// contrast, slightly boosted saturation, NO tonemap rolloff). Master 0.9.
    pub fn toon() -> Self {
        let mut c = Self::off();
        c.intensity = 0.9;
        c.edge.enabled = true;
        c.edge.depth_threshold = 0.12;
        c.edge.strength = 0.75;
        c.grade.enabled = true;
        c.grade.tonemap = false;
        c.grade.contrast = 1.25;
        c.grade.saturation = 1.2;
        c.grade.gamma = 0.95;
        c
    }

    /// A retro-display look: scanlines + aperture mask + ordered dither + a soft
    /// vignette + a muted, phosphor-green grade — for 256-color terminals. Master
    /// 0.7.
    pub fn crt() -> Self {
        let mut c = Self::off();
        c.intensity = 0.7;
        c.scanlines.enabled = true;
        c.scanlines.strength = 0.28;
        c.scanlines.period = 1;
        c.scanlines.mask = 0.12;
        c.dither.enabled = true;
        c.dither.amount = 0.06;
        c.vignette.enabled = true;
        c.vignette.amount = 0.5;
        c.vignette.softness = 0.4;
        c.grade.enabled = true;
        c.grade.tonemap = true;
        c.grade.contrast = 1.1;
        c.grade.saturation = 0.9;
        c.grade.lift = 0.015;
        c.color_temp.enabled = true;
        c.color_temp.tint = 0.12; // a faint phosphor-green push
        c.film_grain.enabled = true;
        c.film_grain.amount = 0.04;
        c
    }

    /// A soft, hazy "dream" look: heavy bloom + DoF + a posterized, warm,
    /// low-contrast grade + chromatic fringe + a gentle vignette. The most stylized
    /// preset. Master 0.85.
    pub fn dream() -> Self {
        let mut c = Self::off();
        c.intensity = 0.85;
        c.bloom.enabled = true;
        c.bloom.threshold = 0.45;
        c.bloom.knee = 0.35;
        c.bloom.radius = 2;
        c.bloom.intensity = 0.9;
        c.dof.enabled = true;
        c.dof.focus_dist = 2.5;
        c.dof.range = 7.0;
        c.dof.max_blur = 0.8;
        c.chromatic.enabled = true;
        c.chromatic.amount = 1.2;
        c.chromatic.edge_only = 0.4;
        c.grade.enabled = true;
        c.grade.tonemap = true;
        c.grade.aces = true;
        c.grade.contrast = 0.92; // soft, low-contrast
        c.grade.saturation = 1.15;
        c.grade.lift = 0.04;
        c.color_temp.enabled = true;
        c.color_temp.temperature = 0.22; // warm amber haze
        c.color_temp.tint = -0.06;
        c.posterize.enabled = true;
        c.posterize.levels = 12;
        c.vignette.enabled = true;
        c.vignette.amount = 0.4;
        c.vignette.softness = 0.55;
        c.film_grain.enabled = true;
        c.film_grain.amount = 0.05;
        c
    }

    /// Builder: set the master mood scalar (clamped to 0..1) and return `self`.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set the film-grain slow-time bucket. The plugin calls this from a SLOW clock
    /// so grain crawls rather than strobing; a static cell at one bucket is byte-
    /// stable. A no-op on the look unless `film_grain.enabled`.
    pub fn with_grain_bucket(mut self, bucket: u32) -> Self {
        self.film_grain.time_bucket = bucket;
        self
    }

    /// Look up a preset by lower-case name (`off`/`clean`/`subtle`/`cinematic`/
    /// `toon`/`crt`/`dream`). Returns `None` for an unknown name so callers can
    /// usage-error.
    pub fn from_preset_name(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::off()),
            "clean" => Some(Self::clean()),
            "subtle" => Some(Self::subtle()),
            "cinematic" => Some(Self::cinematic()),
            "toon" => Some(Self::toon()),
            "crt" => Some(Self::crt()),
            "dream" => Some(Self::dream()),
            _ => None,
        }
    }

    /// All named presets in display order, as `(name, config)` — for a settings UI
    /// or a `--postfx` dump loop that wants to iterate every look.
    pub fn all_presets() -> [(&'static str, Self); 7] {
        [
            ("off", Self::off()),
            ("clean", Self::clean()),
            ("subtle", Self::subtle()),
            ("cinematic", Self::cinematic()),
            ("toon", Self::toon()),
            ("crt", Self::crt()),
            ("dream", Self::dream()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_disables_everything() {
        let c = PostfxConfig::off();
        assert_eq!(c.intensity, 0.0);
        assert!(!c.edge.enabled && !c.ssao.enabled && !c.bloom.enabled);
        assert!(!c.dof.enabled && !c.grade.enabled && !c.vignette.enabled);
        assert!(!c.dither.enabled);
    }

    #[test]
    fn default_is_off() {
        assert_eq!(PostfxConfig::default(), PostfxConfig::off());
    }

    #[test]
    fn presets_enable_effects_and_set_master() {
        assert!(PostfxConfig::subtle().intensity > 0.0);
        assert!(PostfxConfig::cinematic().bloom.enabled);
        assert!(PostfxConfig::cinematic().dof.enabled);
        assert!(PostfxConfig::toon().edge.enabled);
        assert!(PostfxConfig::crt().dither.enabled);
    }

    #[test]
    fn with_intensity_clamps() {
        assert_eq!(PostfxConfig::off().with_intensity(2.0).intensity, 1.0);
        assert_eq!(PostfxConfig::off().with_intensity(-1.0).intensity, 0.0);
        assert_eq!(PostfxConfig::off().with_intensity(0.6).intensity, 0.6);
    }

    #[test]
    fn preset_name_lookup() {
        assert_eq!(
            PostfxConfig::from_preset_name("cinematic"),
            Some(PostfxConfig::cinematic())
        );
        assert_eq!(
            PostfxConfig::from_preset_name("off"),
            Some(PostfxConfig::off())
        );
        assert_eq!(
            PostfxConfig::from_preset_name("clean"),
            Some(PostfxConfig::clean())
        );
        assert_eq!(
            PostfxConfig::from_preset_name("dream"),
            Some(PostfxConfig::dream())
        );
        assert!(PostfxConfig::from_preset_name("nope").is_none());
    }

    #[test]
    fn new_presets_enable_their_effects() {
        assert!(PostfxConfig::clean().grade.enabled && PostfxConfig::clean().film_grain.enabled);
        assert!(PostfxConfig::crt().scanlines.enabled);
        assert!(PostfxConfig::dream().chromatic.enabled && PostfxConfig::dream().posterize.enabled);
        assert!(PostfxConfig::cinematic().color_temp.enabled);
    }

    #[test]
    fn all_presets_round_trip_by_name() {
        for (name, cfg) in PostfxConfig::all_presets() {
            assert_eq!(
                PostfxConfig::from_preset_name(name),
                Some(cfg),
                "preset '{name}'"
            );
        }
    }

    /// The dither-last invariant: `Dither` is the terminal element of the canonical
    /// stage order, and the frozen subsequence ssao→bloom→edge→dof→grade→vignette→
    /// dither appears in that relative order (new stages only slot BETWEEN them).
    #[test]
    fn stage_order_keeps_dither_last_and_frozen_subsequence() {
        let order = PostfxStage::ORDER;
        assert_eq!(
            *order.last().unwrap(),
            PostfxStage::Dither,
            "dither must be last"
        );
        // Frozen subsequence must appear in this relative order.
        let frozen = [
            PostfxStage::Ssao,
            PostfxStage::Bloom,
            PostfxStage::Edge,
            PostfxStage::Dof,
            PostfxStage::Grade,
            PostfxStage::Vignette,
            PostfxStage::Dither,
        ];
        let mut idx = 0usize;
        for &stage in &order {
            if idx < frozen.len() && stage == frozen[idx] {
                idx += 1;
            }
        }
        assert_eq!(
            idx,
            frozen.len(),
            "frozen subsequence must appear in order in ORDER"
        );
    }
}
