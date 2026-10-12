//! The configurable effect set + the [`GlitchConfig`] aggregate and its presets.
//!
//! Each effect is a small `*Effect` struct: an `enabled` flag plus its own
//! params (its own local `intensity`, a `rate`/`speed`, a `seed_off` salt, and
//! where useful a band/region). [`GlitchConfig`] holds one of each plus the
//! master [`GlitchConfig::intensity`] (0..1) that scales every effect at once —
//! the "corruption level" the narrative drives.
//!
//! Effects are deliberately COMBINABLE: enable any subset, and [`apply_glitch`]
//! composes them in a fixed order. Presets ([`GlitchConfig::off`], `subtle`,
//! `heavy`, `terminal_decay`) bundle tasteful combinations.
//!
//! [`apply_glitch`]: crate::apply_glitch

use bevy::prelude::Resource;

pub(crate) mod aberration;
pub(crate) mod bitrot;
pub(crate) mod block;
pub(crate) mod dropout;
pub(crate) mod invert;
pub(crate) mod scanlines;
pub(crate) mod tear;
pub(crate) mod warp;

/// Packed-color helpers shared by the colour effects. `0x00RRGGBB`.
pub(crate) mod color {
    #[inline]
    pub fn unpack(c: u32) -> (u8, u8, u8) {
        (
            ((c >> 16) & 0xFF) as u8,
            ((c >> 8) & 0xFF) as u8,
            (c & 0xFF) as u8,
        )
    }
    #[inline]
    pub fn pack(r: u8, g: u8, b: u8) -> u32 {
        ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
    }
}

// ───────────────────────── per-effect configs ───────────────────────────────

/// **Scanlines / CRT.** Periodically dims every Nth row (the scanline comb) and
/// rolls a brighter/darker horizontal band slowly down the frame (the CRT
/// refresh roll). Pure brightness modulation on fg+bg; no glyph change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanlineEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Row period of the scanline comb (every `period`-th row is dimmed).
    pub period: u32,
    /// How much a scanline row is dimmed (0 = none, 1 = black).
    pub line_dim: f32,
    /// Rows-per-second the bright/dark refresh band rolls down the frame.
    pub roll_speed: f32,
    /// Fraction of the frame height the rolling band spans.
    pub band_frac: f32,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Bit-rot / char corruption.** Replaces a fraction of cells with garbage /
/// mojibake glyphs (and tints them toward a sick green-grey). The corrupt
/// fraction scales with intensity. With `spread > 0` a cell whose neighbour is
/// already corrupt is MORE likely to corrupt too — a contagion / decay feel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BitRotEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Base fraction of cells corrupted at full master intensity (0..1).
    pub density: f32,
    /// Corruption re-rolls this many times per second (flicker rate).
    pub rate: f32,
    /// Extra corruption probability added per already-corrupt 4-neighbour
    /// (0 = no spread; higher = faster contagion).
    pub spread: f32,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Row shift / tearing.** Offsets horizontal slices left/right by a
/// noise-driven amount — datamosh tear lines. Slices are `slice_h` rows tall and
/// each gets one coherent offset (so a band tears as a block, not per-row hash).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TearEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Height (rows) of each independently-shifted slice.
    pub slice_h: u32,
    /// Max horizontal offset (cells) at full intensity.
    pub max_shift: i32,
    /// New tear pattern this many times per second.
    pub rate: f32,
    /// Fraction of slices that actually tear (the rest pass through clean).
    pub tear_frac: f32,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Chromatic aberration / RGB split.** Re-samples the red channel from a cell
/// offset `(+dx, +dy)` and the blue channel from `(-dx, -dy)`, leaving green in
/// place — classic CRT colour fringing. Applied to fg always, and bg when
/// `split_bg` is set. Truecolor makes the red/blue ghosts clearly visible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AberrationEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Column offset (cells) of the red/blue split at full intensity.
    pub dx: i32,
    /// Row offset (cells) of the red/blue split at full intensity.
    pub dy: i32,
    /// Also split the background colour (not just the glyph fg).
    pub split_bg: bool,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Block jitter / displacement.** Picks rectangular regions that jump by a few
/// cells — macroblock corruption. Each block's existence + offset is hashed on a
/// time slice, so blocks pop and move deterministically.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Block width in cells.
    pub block_w: u32,
    /// Block height in cells.
    pub block_h: u32,
    /// Max cell offset a jumping block displaces by, at full intensity.
    pub max_jump: i32,
    /// Fraction of blocks that jump on a given slice (scaled by intensity).
    pub jump_frac: f32,
    /// New block pattern this many times per second.
    pub rate: f32,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Dropout / dead cells.** Intermittently blanks cells to black (signal loss),
/// flickering over time. The blanked fraction scales with intensity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropoutEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Fraction of cells blanked at full master intensity.
    pub density: f32,
    /// New dropout pattern this many times per second (the flicker).
    pub rate: f32,
    /// Glyph a dead cell shows (space by default → a true black hole).
    pub dead_glyph: u32,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Wave / warp.** A per-column vertical sine displacement (and/or per-row
/// horizontal): the signal-distortion wobble. `amp` cells of offset, `freq`
/// waves across the axis, scrolling at `speed`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarpEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Vertical displacement amplitude (cells) per column at full intensity.
    pub amp_y: f32,
    /// Horizontal displacement amplitude (cells) per row at full intensity.
    pub amp_x: f32,
    /// Spatial frequency (waves across the frame).
    pub freq: f32,
    /// Scroll speed of the wave (cycles per second).
    pub speed: f32,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Invert / posterize flashes.** On a pulsing schedule, inverts colours and/or
/// quantizes (posterizes) them — a brief banded or full-frame corruption flash.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvertEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// Pulses per second; each pulse fires for `duty` of its period.
    pub rate: f32,
    /// Fraction of each pulse period the flash is ON (0..1).
    pub duty: f32,
    /// Posterize level: colours quantized to this many steps per channel during
    /// a flash (`0` or `>=256` → no posterize, pure invert).
    pub levels: u32,
    /// If `>0`, only a horizontal BAND of this many rows flashes (centred on a
    /// time-scrolled position); `0` = the whole frame flashes.
    pub band_h: u32,
    /// Per-effect seed salt (offsets the shared frame hash so effects don't
    /// all corrupt the same cells in lockstep).
    pub seed_off: u64,
}

/// **Phosphor ghosting / persistence.** Blends a dimmed copy of the PREVIOUS
/// presented frame under the current one — faint decay trails. Stateful (needs a
/// prior-frame buffer), so the pure [`apply_glitch`](crate::apply_glitch) does
/// NOT apply it; the [`GlitchPlugin`](crate::GlitchPlugin) drives it from its
/// snapshot history. Kept in the config so a level can request it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhosphorEffect {
    /// When `false`, this effect is skipped entirely (a no-op stage).
    pub enabled: bool,
    /// Local strength 0..1, multiplied by the master intensity.
    pub intensity: f32,
    /// How strongly the previous frame bleeds through (0 = none, 1 = full).
    pub persistence: f32,
}

// ───────────────────────────── aggregate ────────────────────────────────────

/// The full glitch configuration: one of every effect plus the master
/// [`intensity`](Self::intensity) (the "corruption level" 0..1) that scales them
/// all. Exposed as a Bevy [`Resource`] so the game can set it per-level, by
/// narrative corruption stage, or from a settings toggle.
///
/// Default is [`GlitchConfig::off`] — the identity — so shipped levels are
/// unaffected unless something opts in.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct GlitchConfig {
    /// Master corruption scalar 0..1. `0` = identity (frame untouched); `1` =
    /// each effect at its authored full strength. Every effect multiplies its
    /// own local `intensity` by this, so one knob drives the whole decay.
    pub intensity: f32,
    /// Scanline / CRT refresh-roll pass (see [`ScanlineEffect`]).
    pub scanlines: ScanlineEffect,
    /// Bit-rot / character corruption pass (see [`BitRotEffect`]).
    pub bitrot: BitRotEffect,
    /// Row shift / tearing pass (see [`TearEffect`]).
    pub tear: TearEffect,
    /// Chromatic aberration / RGB split pass (see [`AberrationEffect`]).
    pub aberration: AberrationEffect,
    /// Block jitter / displacement pass (see [`BlockEffect`]).
    pub block: BlockEffect,
    /// Dropout / dead-cell pass (see [`DropoutEffect`]).
    pub dropout: DropoutEffect,
    /// Wave / warp displacement pass (see [`WarpEffect`]).
    pub warp: WarpEffect,
    /// Invert / posterize flash pass (see [`InvertEffect`]).
    pub invert: InvertEffect,
    /// Phosphor ghosting / persistence pass (see [`PhosphorEffect`]). Stateful;
    /// only the plugin applies it, not the pure [`crate::apply_glitch`].
    pub phosphor: PhosphorEffect,
}

impl Default for GlitchConfig {
    /// The default is OFF — additive guarantee for shipped levels.
    fn default() -> Self {
        Self::off()
    }
}

impl GlitchConfig {
    /// All effects disabled, master intensity 0. [`apply_glitch`] is the identity
    /// on this config (the frame is left byte-identical).
    ///
    /// [`apply_glitch`]: crate::apply_glitch
    pub fn off() -> Self {
        GlitchConfig {
            intensity: 0.0,
            scanlines: ScanlineEffect {
                enabled: false,
                intensity: 1.0,
                period: 3,
                line_dim: 0.35,
                roll_speed: 8.0,
                band_frac: 0.18,
                seed_off: 0x01,
            },
            bitrot: BitRotEffect {
                enabled: false,
                intensity: 1.0,
                density: 0.06,
                rate: 12.0,
                spread: 0.25,
                seed_off: 0x02,
            },
            tear: TearEffect {
                enabled: false,
                intensity: 1.0,
                slice_h: 2,
                max_shift: 6,
                rate: 10.0,
                tear_frac: 0.4,
                seed_off: 0x03,
            },
            aberration: AberrationEffect {
                enabled: false,
                intensity: 1.0,
                dx: 1,
                dy: 0,
                split_bg: true,
                seed_off: 0x04,
            },
            block: BlockEffect {
                enabled: false,
                intensity: 1.0,
                block_w: 6,
                block_h: 3,
                max_jump: 3,
                jump_frac: 0.25,
                rate: 6.0,
                seed_off: 0x05,
            },
            dropout: DropoutEffect {
                enabled: false,
                intensity: 1.0,
                density: 0.04,
                rate: 15.0,
                dead_glyph: u32::from(' '),
                seed_off: 0x06,
            },
            warp: WarpEffect {
                enabled: false,
                intensity: 1.0,
                amp_y: 1.5,
                amp_x: 0.0,
                freq: 2.5,
                speed: 0.6,
                seed_off: 0x07,
            },
            invert: InvertEffect {
                enabled: false,
                intensity: 1.0,
                rate: 0.7,
                duty: 0.12,
                levels: 0,
                band_h: 0,
                seed_off: 0x08,
            },
            phosphor: PhosphorEffect {
                enabled: false,
                intensity: 1.0,
                persistence: 0.4,
            },
        }
    }

    /// A light, mostly-cosmetic preset: gentle scanlines + a faint chromatic
    /// fringe + a slow warp wobble. Reads as "old CRT", barely-degraded. Master
    /// intensity 0.35.
    pub fn subtle() -> Self {
        let mut c = Self::off();
        c.intensity = 0.35;
        c.scanlines.enabled = true;
        c.scanlines.line_dim = 0.22;
        c.aberration.enabled = true;
        c.aberration.dx = 1;
        c.warp.enabled = true;
        c.warp.amp_y = 0.8;
        c.phosphor.enabled = true;
        c.phosphor.persistence = 0.25;
        c
    }

    /// A loud, obviously-broken preset: every spatial + colour effect on, strong.
    /// Tearing, block jitter, RGB split, bit-rot, dropout, invert pulses. Master
    /// intensity 0.85.
    pub fn heavy() -> Self {
        let mut c = Self::off();
        c.intensity = 0.85;
        c.scanlines.enabled = true;
        c.bitrot.enabled = true;
        c.bitrot.density = 0.10;
        c.tear.enabled = true;
        c.aberration.enabled = true;
        c.aberration.dx = 2;
        c.aberration.dy = 1;
        c.block.enabled = true;
        c.dropout.enabled = true;
        c.warp.enabled = true;
        c.invert.enabled = true;
        c.phosphor.enabled = true;
        c
    }

    /// A "dying terminal" preset: a look weighted toward spreading bit-rot +
    /// dropout + scanlines + a faint phosphor trail — suited to driving off a
    /// host game's own narrative corruption timer. Pass the narrative
    /// corruption level as the master via [`with_intensity`](Self::with_intensity).
    pub fn terminal_decay() -> Self {
        let mut c = Self::off();
        c.intensity = 0.5;
        c.scanlines.enabled = true;
        c.scanlines.line_dim = 0.3;
        c.bitrot.enabled = true;
        c.bitrot.density = 0.08;
        c.bitrot.spread = 0.5; // contagion: corruption spreads outward
        c.dropout.enabled = true;
        c.dropout.density = 0.05;
        c.aberration.enabled = true;
        c.tear.enabled = true;
        c.tear.tear_frac = 0.25;
        c.phosphor.enabled = true;
        c.phosphor.persistence = 0.35;
        c
    }

    /// Builder: set the master corruption scalar (clamped to 0..1) and return
    /// `self`. The natural way to drive a preset by narrative corruption level:
    /// `GlitchConfig::terminal_decay().with_intensity(level)`.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_disables_everything() {
        let c = GlitchConfig::off();
        assert_eq!(c.intensity, 0.0);
        assert!(!c.scanlines.enabled && !c.bitrot.enabled && !c.tear.enabled);
        assert!(!c.aberration.enabled && !c.block.enabled && !c.dropout.enabled);
        assert!(!c.warp.enabled && !c.invert.enabled && !c.phosphor.enabled);
    }

    #[test]
    fn default_is_off() {
        assert_eq!(GlitchConfig::default(), GlitchConfig::off());
    }

    #[test]
    fn presets_enable_effects_and_set_master() {
        assert!(GlitchConfig::subtle().intensity > 0.0);
        assert!(GlitchConfig::heavy().intensity > GlitchConfig::subtle().intensity);
        assert!(GlitchConfig::terminal_decay().bitrot.spread > 0.0);
    }

    #[test]
    fn with_intensity_clamps() {
        assert_eq!(GlitchConfig::off().with_intensity(2.0).intensity, 1.0);
        assert_eq!(GlitchConfig::off().with_intensity(-1.0).intensity, 0.0);
        assert_eq!(GlitchConfig::off().with_intensity(0.6).intensity, 0.6);
    }
}
