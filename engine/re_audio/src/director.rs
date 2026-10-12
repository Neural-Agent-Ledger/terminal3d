//! `director` — generative arrangement. **Pure logic, seeded RNG, no device.**
//!
//! On each beat tick the director decides what (if anything) plays, drawing from
//! a per-theme palette: sparse tonal notes and drone accents on strong beats,
//! terminal glitches on weak beats, all at low density so the soundscape evolves
//! rather than loops verbatim.
//!
//! Determinism: the director carries **its own** LCG (the same Numerical-Recipes
//! pattern as [`crate::synth`]), never `rand` and never the wall clock, so a
//! given seed produces a byte-for-byte reproducible event stream. That is what
//! the reproducibility tests rely on. It only *proposes* one-shots; the plugin
//! still runs them through [`crate::policy`] + [`crate::transport`] to pick the
//! exact scheduled tick.

use crate::synth::SoundId;
use crate::transport::{BeatClass, Grid};

/// One scheduling proposal from the director: which sound to play, plus a pan
/// hint in `[-1.0, 1.0]` (hard-left .. hard-right) for stereo spread. The plugin
/// resolves the actual scheduled tick; the director only chooses *what*.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectorEvent {
    /// The sound to play.
    pub sound: SoundId,
    /// Stereo pan hint, `[-1.0, 1.0]`.
    pub pan: f32,
}

/// The musical theme. Selects the director's density/palette and (in
/// `playback`) the master-effect character.
///
/// [`Theme::Default`], [`Theme::Cold`] and [`Theme::Garden`] are the shipped
/// three and are **unchanged** — levels already reference them. `P1`..`P4` are
/// the four campaign *movements* (mapped from the campaign manifest's movement
/// tag). Each is a mood, never a narrator: MNEMOS and LETHE stay ambient
/// forces, so no movement introduces a voice-like lead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Theme {
    /// Balanced default.
    #[default]
    Default,
    /// Sparser, colder, darker — the constellation levels.
    Cold,
    /// Warmer, a touch busier — the garden levels.
    Garden,
    /// **Movement 1 — "first light".** Warm and sparse: gentle low-mid
    /// pentatonic notes, almost no glitch, slow density. Innocence: the system
    /// has not started watching yet.
    P1FirstLight,
    /// **Movement 2 — "the gaze".** A steady, slightly detuned soft pulse
    /// ([`SoundId::GazePulse`]) under the bed, with sparse mid-register notes in
    /// the gaps. Recognition: something keeps time with you.
    P2TheGaze,
    /// **Movement 3 — "grief" (LETHE).** Hollow: the filtered river wash
    /// ([`SoundId::RiverWash`]) as a bed-like swell, low minor notes, long
    /// decays, thicker drone accents.
    P3Grief,
    /// **Movement 4 — "closing".** Near-silence plus a heartbeat: a soft low
    /// [`SoundId::HeartLub`] on the downbeat answered by [`SoundId::HeartDub`]
    /// on the next beat, and almost nothing else. The choice.
    P4Closing,
}

impl Theme {
    /// Every theme, in declaration order. Used by the exhaustiveness tests and
    /// by callers that want to enumerate the palette (e.g. a debug menu).
    pub const ALL: [Theme; 7] = [
        Theme::Default,
        Theme::Cold,
        Theme::Garden,
        Theme::P1FirstLight,
        Theme::P2TheGaze,
        Theme::P3Grief,
        Theme::P4Closing,
    ];

    /// The four campaign movements, in order `P1..P4`.
    pub const MOVEMENTS: [Theme; 4] = [
        Theme::P1FirstLight,
        Theme::P2TheGaze,
        Theme::P3Grief,
        Theme::P4Closing,
    ];
}

// ── tonal windows ───────────────────────────────────────────────────────────
//
// Each theme draws its pitched notes from a *window* into
// `SoundId::TONAL_PALETTE` (A minor pentatonic, low → high) rather than the
// whole span. That is what makes a movement sound like itself: the register is
// the character, and every window stays inside the one scale so no movement can
// modulate into something cheerful.

/// `P1FirstLight` — warm low/mid: A3 C4 D4 E4 G4.
const TONAL_WARM: [SoundId; 5] = [
    SoundId::ToneA3,
    SoundId::ToneC4,
    SoundId::ToneD4,
    SoundId::ToneE4,
    SoundId::ToneG4,
];

/// `P2TheGaze` — mid register, above the pulse: D4 E4 G4 A4.
const TONAL_MID: [SoundId; 4] = [
    SoundId::ToneD4,
    SoundId::ToneE4,
    SoundId::ToneG4,
    SoundId::ToneA4,
];

/// `P3Grief` — the hollow bottom of the scale: A3 C4 D4.
const TONAL_LOW: [SoundId; 3] = [SoundId::ToneA3, SoundId::ToneC4, SoundId::ToneD4];

/// `P4Closing` — two notes left: A3 C4.
const TONAL_CLOSING: [SoundId; 2] = [SoundId::ToneA3, SoundId::ToneC4];

/// Per-theme generative weights. Probabilities are in `[0.0, 1.0]` and are
/// evaluated independently per eligible beat.
///
/// A theme has two kinds of material:
/// * the **generic** layer — a tonal note or a drone accent on a strong beat, a
///   glitch on a weak beat (the shipped behaviour, unchanged for
///   `Default`/`Cold`/`Garden`), and
/// * an optional **motif** — one sound that *is* the movement (the gaze pulse,
///   the river wash, the heartbeat pair). The motif is checked first and
///   replaces the generic pick on that beat, so raising a motif probability
///   makes the movement more itself rather than simply busier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectorConfig {
    /// Chance of emitting a tonal note on a strong beat.
    pub strong_tone_prob: f64,
    /// Chance of emitting a drone accent on a strong beat (checked only if no
    /// tonal note fired this beat).
    pub strong_drone_prob: f64,
    /// Chance of emitting a glitch on a weak beat.
    pub weak_glitch_prob: f64,
    /// The theme's downbeat motif, if any. Checked before the tonal/drone pick.
    pub strong_motif: Option<SoundId>,
    /// Chance the [`strong_motif`](Self::strong_motif) fires on a strong beat.
    pub strong_motif_prob: f64,
    /// The theme's weak-beat motif, if any (the answering half of a pair).
    /// Checked before the glitch pick.
    pub weak_motif: Option<SoundId>,
    /// Chance the [`weak_motif`](Self::weak_motif) fires on an eligible weak beat.
    pub weak_motif_prob: f64,
    /// When `Some(n)`, the [`weak_motif`](Self::weak_motif) may fire **only** on
    /// beat `n` of the bar (`0` = downbeat, so a "next beat" answer is `1`).
    /// `None` = any weak beat is eligible.
    pub weak_motif_beat: Option<u64>,
    /// The window of [`SoundId::TONAL_PALETTE`] this theme draws notes from.
    pub tonal: &'static [SoundId],
    /// Stereo-width multiplier applied to the emitted pan, in `[0.0, 1.0]`.
    /// Low values keep a movement centred (the heartbeat is *inside* you).
    pub pan_spread: f32,
}

impl DirectorConfig {
    /// The neutral base every theme is written as a diff against: the shipped
    /// generic layer with no motif and the full tonal palette.
    const fn base(strong_tone_prob: f64, strong_drone_prob: f64, weak_glitch_prob: f64) -> Self {
        DirectorConfig {
            strong_tone_prob,
            strong_drone_prob,
            weak_glitch_prob,
            strong_motif: None,
            strong_motif_prob: 0.0,
            weak_motif: None,
            weak_motif_prob: 0.0,
            weak_motif_beat: None,
            tonal: &SoundId::TONAL_PALETTE,
            pan_spread: 1.0,
        }
    }

    /// The config for a theme.
    pub fn for_theme(theme: Theme) -> Self {
        match theme {
            Theme::Default => DirectorConfig::base(0.25, 0.10, 0.30),
            // Cold: rarer everything, sparse and spacious.
            Theme::Cold => DirectorConfig::base(0.18, 0.06, 0.18),
            // Garden: warmer, a little more melodic, slightly busier.
            Theme::Garden => DirectorConfig::base(0.35, 0.12, 0.28),

            // ── the four movements ──────────────────────────────────────────
            // P1 "first light": warm and sparse. Notes are the only real
            // material; glitches are almost absent (nothing is corrupt yet) and
            // the stereo image is narrow-ish and intimate. Note the glitch
            // probability is per *weak* beat, of which there are three for every
            // downbeat — 0.03 is what "you barely ever hear one" costs.
            Theme::P1FirstLight => DirectorConfig {
                tonal: &TONAL_WARM,
                pan_spread: 0.8,
                ..DirectorConfig::base(0.22, 0.05, 0.03)
            },
            // P2 "the gaze": the pulse carries most downbeats; the leftover
            // downbeats get a mid-register note, and a thin glitch layer keeps
            // the watching from feeling safe. Narrow image — it watches from one
            // place.
            Theme::P2TheGaze => DirectorConfig {
                strong_motif: Some(SoundId::GazePulse),
                strong_motif_prob: 0.80,
                tonal: &TONAL_MID,
                pan_spread: 0.35,
                ..DirectorConfig::base(0.30, 0.10, 0.14)
            },
            // P3 "grief": the river takes a bit under half the downbeats, the
            // rest are low hollow notes and (more often than elsewhere) drone
            // accents. Wide, washed-out image.
            Theme::P3Grief => DirectorConfig {
                strong_motif: Some(SoundId::RiverWash),
                strong_motif_prob: 0.45,
                tonal: &TONAL_LOW,
                pan_spread: 1.0,
                ..DirectorConfig::base(0.34, 0.16, 0.10)
            },
            // P4 "closing": near-silence plus a heartbeat. The lub lands on the
            // downbeat, the dub on the beat right after it (`weak_motif_beat`),
            // and both may falter — a heart that skips. Everything else is
            // almost gone; the pair sits centred.
            Theme::P4Closing => DirectorConfig {
                strong_motif: Some(SoundId::HeartLub),
                strong_motif_prob: 0.85,
                weak_motif: Some(SoundId::HeartDub),
                weak_motif_prob: 0.75,
                weak_motif_beat: Some(1),
                tonal: &TONAL_CLOSING,
                pan_spread: 0.2,
                ..DirectorConfig::base(0.06, 0.04, 0.02)
            },
        }
    }
}

/// Which beat of the bar `tick` falls on (`0` = the downbeat). Only meaningful
/// for ticks that are beat boundaries; off-beat subdivisions are never passed
/// here (they are rejected by [`Grid::classify`] first).
#[inline]
fn beat_in_bar(tick: u64, grid: Grid) -> u64 {
    (tick % grid.ticks_per_bar()) / grid.ticks_per_beat()
}

/// 32-bit LCG (Numerical Recipes constants), copied from [`crate::synth`] so the
/// director's randomness is self-contained and reproducible. Not `pub`: the
/// director's stream is an internal detail pinned by its seed.
#[derive(Clone, Copy, Debug)]
struct Lcg(u32);

impl Lcg {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    /// Next value in `[0.0, 1.0)` from the top 24 bits.
    #[inline]
    fn next_unit(&mut self) -> f64 {
        (self.next_u32() >> 8) as f64 / 16_777_216.0
    }
}

/// The generative arranger. Advance it one beat at a time with [`Director::on_beat`].
///
/// Derives Bevy's `Resource` purely so [`crate::plugin`] can hold it; the type's
/// logic is engine-agnostic and unit-tested with no Bevy `App`.
#[derive(Clone, Debug, bevy::prelude::Resource)]
pub struct Director {
    rng: Lcg,
    config: DirectorConfig,
    theme: Theme,
    /// Seconds of "solve silence" left on the theme layer. `0.0` = playing.
    silence_secs: f32,
}

impl Director {
    /// Longest solve-silence window the director will honour, in seconds. A
    /// request is clamped to this: a bug in the caller must not be able to mute
    /// the score for the rest of the level.
    pub const MAX_SILENCE_SECS: f32 = 8.0;

    /// The designed solve-silence window is 3-5 s; this is the midpoint the
    /// game sends on a peak solve unless it overrides.
    pub const SOLVE_SILENCE_SECS: f32 = 4.0;

    /// Create a director for `theme`, seeded for reproducibility.
    pub fn new(theme: Theme, seed: u32) -> Self {
        Director {
            // Avoid a zero seed degenerating the LCG's early values.
            rng: Lcg(seed | 1),
            config: DirectorConfig::for_theme(theme),
            theme,
            silence_secs: 0.0,
        }
    }

    /// The director's current theme.
    #[inline]
    pub fn theme(&self) -> Theme {
        self.theme
    }

    /// The director's active generative weights (the theme's config).
    #[inline]
    pub fn config(&self) -> DirectorConfig {
        self.config
    }

    /// Reconfigure to a new theme. Keeps the current RNG state so a live theme
    /// swap does not reset the (deterministic) stream mid-level. Also keeps any
    /// in-flight solve silence — a level change during the held breath does not
    /// cancel it.
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
        self.config = DirectorConfig::for_theme(theme);
    }

    // ── solve silence ("the music holds its breath") ─────────────────────────
    //
    // On the game's peak solves, the *theme* layer goes quiet for a few
    // seconds. Only this layer: the ambience beds keep running
    // (they are `playback`'s crossfaded loops, which the director never
    // touches), and one-shots the game triggers itself — footsteps, UI, story
    // stingers — are unaffected, because they never pass through here. The world
    // does not die; the music stops breathing for a moment.

    /// Mute the generative theme layer for `seconds`.
    ///
    /// **Re-entrant by extension, never by stacking**: calling it again while a
    /// window is open sets the remaining time to `max(remaining, seconds)`, so
    /// two solves 1 s apart cannot add up to a 10 s hole. `seconds` is clamped
    /// into `[0, MAX_SILENCE_SECS]`; a non-finite or negative request is a no-op.
    pub fn hold_breath(&mut self, seconds: f32) {
        if !seconds.is_finite() || seconds <= 0.0 {
            return;
        }
        let want = seconds.min(Self::MAX_SILENCE_SECS);
        if want > self.silence_secs {
            self.silence_secs = want;
        }
    }

    /// Advance the solve-silence window by `dt` seconds (called once per frame
    /// from [`crate::plugin`]). Saturates at zero; a non-finite `dt` is ignored.
    pub fn advance_silence(&mut self, dt: f32) {
        if self.silence_secs <= 0.0 || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.silence_secs = (self.silence_secs - dt).max(0.0);
    }

    /// Seconds of solve silence still to run (`0.0` when the theme is playing).
    #[inline]
    pub fn silence_remaining(&self) -> f32 {
        self.silence_secs
    }

    /// True while the theme layer is muted by a solve-silence window.
    #[inline]
    pub fn is_silent(&self) -> bool {
        self.silence_secs > 0.0
    }

    /// Drop any open solve-silence window immediately (e.g. leaving the level).
    #[inline]
    pub fn release_breath(&mut self) {
        self.silence_secs = 0.0;
    }

    /// Decide what (if anything) plays on the beat at absolute `tick`.
    ///
    /// Off-beat subdivisions (ticks that are neither strong nor weak under
    /// `grid`) never produce an event. On a strong beat the director may emit
    /// the theme's downbeat motif, else a sparse tonal note, else a drone
    /// accent; on a weak beat it may emit the theme's answering motif, else a
    /// glitch. At most one event per beat.
    ///
    /// While a solve-silence window is open ([`Director::is_silent`]) the
    /// arrangement is still *computed* and the RNG still advances — the event is
    /// simply dropped. So the held breath removes a few seconds of theme audio
    /// without shifting the stream that follows it: the music resumes exactly
    /// where it would have been.
    ///
    /// The RNG advances by a fixed four draws per classified tick regardless of
    /// outcome, so the stream stays reproducible.
    pub fn on_beat(&mut self, tick: u64, grid: Grid) -> Option<DirectorEvent> {
        let proposed = self.propose(tick, grid);
        if self.is_silent() {
            return None;
        }
        proposed
    }

    /// The arrangement decision, ignoring solve silence. Always advances the RNG
    /// by exactly four draws for a beat-boundary tick, zero otherwise.
    fn propose(&mut self, tick: u64, grid: Grid) -> Option<DirectorEvent> {
        let class = grid.classify(tick)?;

        // Draw every decision value up front: a fixed number of draws per call
        // keeps the stream independent of which branch is taken.
        let roll = self.rng.next_unit();
        let pan_unit = self.rng.next_unit();
        let motif_roll = self.rng.next_unit();
        let idx = self.rng.next_u32() as usize;

        let cfg = self.config;
        let spread = cfg.pan_spread.clamp(0.0, 1.0);
        let pan = ((pan_unit as f32) * 2.0 - 1.0) * spread;

        match class {
            BeatClass::Strong => {
                // The movement's own motif first: it *is* the theme.
                if let Some(motif) = cfg.strong_motif {
                    if motif_roll < cfg.strong_motif_prob {
                        return Some(DirectorEvent {
                            sound: motif,
                            // Motifs sit closer to centre than free material.
                            pan: pan * 0.5,
                        });
                    }
                }
                if roll < cfg.strong_tone_prob {
                    let palette = cfg.tonal;
                    if palette.is_empty() {
                        return None;
                    }
                    Some(DirectorEvent {
                        sound: palette[idx % palette.len()],
                        pan,
                    })
                } else if roll < cfg.strong_tone_prob + cfg.strong_drone_prob {
                    // A drone accent, panned gently.
                    let drones = [SoundId::DroneLayerA, SoundId::DroneLayerB];
                    Some(DirectorEvent {
                        sound: drones[idx % drones.len()],
                        pan: pan * 0.5,
                    })
                } else {
                    None
                }
            }
            BeatClass::Weak => {
                if let Some(motif) = cfg.weak_motif {
                    let beat_ok = cfg
                        .weak_motif_beat
                        .is_none_or(|n| beat_in_bar(tick, grid) == n);
                    if beat_ok && motif_roll < cfg.weak_motif_prob {
                        return Some(DirectorEvent {
                            sound: motif,
                            pan: pan * 0.5,
                        });
                    }
                }
                if roll < cfg.weak_glitch_prob {
                    let fam = SoundId::GLITCH_FAMILY;
                    Some(DirectorEvent {
                        sound: fam[idx % fam.len()],
                        pan,
                    })
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: Grid = Grid {
        beats_per_bar: 4,
        subdivision: 2,
    };

    fn run(seed: u32, theme: Theme, ticks: u64) -> Vec<(u64, DirectorEvent)> {
        let mut d = Director::new(theme, seed);
        let mut out = Vec::new();
        for t in 0..ticks {
            if let Some(ev) = d.on_beat(t, G) {
                out.push((t, ev));
            }
        }
        out
    }

    #[test]
    fn same_seed_same_stream() {
        let a = run(0xBEEF, Theme::Default, 400);
        let b = run(0xBEEF, Theme::Default, 400);
        assert_eq!(a, b, "director not reproducible for a fixed seed");
        // And it actually produced *something* (not a silent stub).
        assert!(!a.is_empty(), "director never emitted any event");
    }

    #[test]
    fn different_seeds_differ() {
        let a = run(0x1111, Theme::Default, 400);
        let b = run(0x2222, Theme::Default, 400);
        assert_ne!(a, b, "different seeds gave identical streams");
    }

    #[test]
    fn off_beats_never_emit() {
        // Only beat-boundary ticks may emit; off-beat subdivisions never do.
        let mut d = Director::new(Theme::Garden, 0xABCD);
        for t in 0..400 {
            let ev = d.on_beat(t, G);
            if G.classify(t).is_none() {
                assert!(ev.is_none(), "off-beat tick {t} emitted {ev:?}");
            }
        }
    }

    #[test]
    fn events_respect_class_palettes() {
        // Strong beats only ever emit tonal/drone sounds; weak beats only ever
        // emit glitch-family sounds.
        let mut d = Director::new(Theme::Default, 0x55AA);
        for t in 0..600 {
            if let Some(ev) = d.on_beat(t, G) {
                match G.classify(t).unwrap() {
                    BeatClass::Strong => {
                        let tonal = SoundId::TONAL_PALETTE.contains(&ev.sound);
                        let drone = matches!(ev.sound, SoundId::DroneLayerA | SoundId::DroneLayerB);
                        assert!(tonal || drone, "strong beat {t} emitted {:?}", ev.sound);
                    }
                    BeatClass::Weak => {
                        assert!(
                            SoundId::GLITCH_FAMILY.contains(&ev.sound),
                            "weak beat {t} emitted {:?}",
                            ev.sound
                        );
                    }
                }
                assert!((-1.0..=1.0).contains(&ev.pan));
            }
        }
    }

    #[test]
    fn density_within_bounds() {
        // Over a long run the emit rate must be low (sparse), well under one
        // event per beat — proving it "evolves rather than loops" and never
        // floods. Count eligible beats vs emitted events.
        let mut d = Director::new(Theme::Default, 0x0DDF);
        let ticks = 2000u64;
        let mut beats = 0usize;
        let mut events = 0usize;
        for t in 0..ticks {
            if G.classify(t).is_some() {
                beats += 1;
                if d.on_beat(t, G).is_some() {
                    events += 1;
                }
            } else {
                // still must advance the director on off-beats (returns None)
                let _ = d.on_beat(t, G);
            }
        }
        let rate = events as f64 / beats as f64;
        assert!(
            (0.05..0.6).contains(&rate),
            "emit rate {rate} outside sparse bounds (events={events}, beats={beats})"
        );
    }

    // ───────────────────── the four campaign movements ───────────────────────

    /// Eligible beats and emitted events for a theme over `ticks` ticks.
    fn stats(seed: u32, theme: Theme, ticks: u64) -> (usize, Vec<(u64, DirectorEvent)>) {
        let mut d = Director::new(theme, seed);
        let mut beats = 0usize;
        let mut out = Vec::new();
        for t in 0..ticks {
            if G.classify(t).is_some() {
                beats += 1;
            }
            if let Some(ev) = d.on_beat(t, G) {
                out.push((t, ev));
            }
        }
        (beats, out)
    }

    #[test]
    fn every_theme_has_a_config_and_a_tonal_window_inside_the_scale() {
        assert_eq!(Theme::ALL.len(), 7);
        for theme in Theme::ALL {
            let cfg = DirectorConfig::for_theme(theme);
            assert!(!cfg.tonal.is_empty(), "{theme:?} has an empty tonal window");
            for id in cfg.tonal {
                assert!(
                    SoundId::TONAL_PALETTE.contains(id),
                    "{theme:?} draws {} from outside the pentatonic palette",
                    id.name()
                );
            }
            assert!(
                (0.0..=1.0).contains(&cfg.pan_spread),
                "{theme:?} pan_spread"
            );
            for p in [
                cfg.strong_tone_prob,
                cfg.strong_drone_prob,
                cfg.weak_glitch_prob,
                cfg.strong_motif_prob,
                cfg.weak_motif_prob,
            ] {
                assert!((0.0..=1.0).contains(&p), "{theme:?} probability {p}");
            }
            // A motif probability without a motif (or vice versa) is a typo.
            assert_eq!(
                cfg.strong_motif.is_some(),
                cfg.strong_motif_prob > 0.0,
                "{theme:?} strong motif/probability disagree"
            );
            assert_eq!(
                cfg.weak_motif.is_some(),
                cfg.weak_motif_prob > 0.0,
                "{theme:?} weak motif/probability disagree"
            );
        }
    }

    #[test]
    fn shipped_three_themes_are_untouched() {
        // Default/Cold/Garden are referenced by shipped levels: their weights
        // must stay exactly as they were, and they must have no motif layer.
        let expected = [
            (Theme::Default, 0.25, 0.10, 0.30),
            (Theme::Cold, 0.18, 0.06, 0.18),
            (Theme::Garden, 0.35, 0.12, 0.28),
        ];
        for (theme, tone, drone, glitch) in expected {
            let c = DirectorConfig::for_theme(theme);
            assert_eq!(c.strong_tone_prob, tone, "{theme:?}");
            assert_eq!(c.strong_drone_prob, drone, "{theme:?}");
            assert_eq!(c.weak_glitch_prob, glitch, "{theme:?}");
            assert_eq!(c.strong_motif, None, "{theme:?} grew a motif");
            assert_eq!(c.weak_motif, None, "{theme:?} grew a motif");
            assert_eq!(c.tonal, &SoundId::TONAL_PALETTE, "{theme:?} tonal window");
            assert_eq!(c.pan_spread, 1.0, "{theme:?} pan spread");
        }
    }

    #[test]
    fn movements_emit_only_their_own_material() {
        // Whatever a movement emits must be one of: its tonal window, a drone
        // accent, the glitch family, or its own motifs. Nothing else — in
        // particular no stinger and no other movement's motif (canon guard: the
        // movements are moods, they never borrow each other's voice).
        for theme in Theme::MOVEMENTS {
            let cfg = DirectorConfig::for_theme(theme);
            let (_, events) = stats(0x4711, theme, 4000);
            assert!(!events.is_empty(), "{theme:?} emitted nothing at all");
            for (t, ev) in &events {
                let ok = cfg.tonal.contains(&ev.sound)
                    || matches!(ev.sound, SoundId::DroneLayerA | SoundId::DroneLayerB)
                    || SoundId::GLITCH_FAMILY.contains(&ev.sound)
                    || cfg.strong_motif == Some(ev.sound)
                    || cfg.weak_motif == Some(ev.sound);
                assert!(ok, "{theme:?} emitted {:?} at tick {t}", ev.sound);
                assert!(
                    !SoundId::STINGERS.contains(&ev.sound),
                    "{theme:?} fired the story stinger {:?}",
                    ev.sound
                );
                assert!((-1.0..=1.0).contains(&ev.pan));
                assert!(
                    ev.pan.abs() <= cfg.pan_spread + 1e-6,
                    "{theme:?} pan too wide"
                );
                // Class discipline holds for the movements too.
                match G.classify(*t).unwrap() {
                    BeatClass::Strong => assert_ne!(Some(ev.sound), cfg.weak_motif),
                    BeatClass::Weak => assert_ne!(Some(ev.sound), cfg.strong_motif),
                }
            }
        }
    }

    #[test]
    fn p1_first_light_is_warm_sparse_and_almost_glitch_free() {
        let (beats, events) = stats(0xF1A7, Theme::P1FirstLight, 4000);
        let glitches = events
            .iter()
            .filter(|(_, e)| SoundId::GLITCH_FAMILY.contains(&e.sound))
            .count();
        let tones = events
            .iter()
            .filter(|(_, e)| SoundId::TONAL_PALETTE.contains(&e.sound))
            .count();
        // Innocence: notes, hardly any corruption.
        assert!(
            tones > glitches * 2,
            "P1 should be tonal, not glitchy (tones={tones}, glitches={glitches})"
        );
        let glitch_rate = glitches as f64 / beats as f64;
        assert!(glitch_rate < 0.06, "P1 glitch rate {glitch_rate} too busy");
        // Sparse overall.
        let rate = events.len() as f64 / beats as f64;
        assert!(
            (0.02..0.20).contains(&rate),
            "P1 emit rate {rate} not sparse"
        );
        // Warm register only: never the top of the palette.
        for (_, e) in &events {
            assert_ne!(e.sound, SoundId::ToneC5);
            assert_ne!(e.sound, SoundId::ToneD5);
        }
    }

    #[test]
    fn p2_the_gaze_pulses_steadily_on_downbeats() {
        let (_, events) = stats(0x6A2E, Theme::P2TheGaze, 4000);
        let downbeats = (0..4000u64)
            .filter(|t| G.classify(*t) == Some(BeatClass::Strong))
            .count();
        let pulses = events
            .iter()
            .filter(|(_, e)| e.sound == SoundId::GazePulse)
            .count();
        let pulse_rate = pulses as f64 / downbeats as f64;
        assert!(
            (0.6..=1.0).contains(&pulse_rate),
            "P2 pulse rate {pulse_rate} — the gaze must be *steady*"
        );
        // The pulse is a downbeat motif, never a weak-beat event.
        for (t, e) in &events {
            if e.sound == SoundId::GazePulse {
                assert_eq!(G.classify(*t), Some(BeatClass::Strong));
                // Centred: the watcher sits in one place.
                assert!(e.pan.abs() <= 0.2, "gaze pulse panned to {}", e.pan);
            }
        }
    }

    #[test]
    fn p3_grief_leans_on_the_river_and_the_low_register() {
        let (_, events) = stats(0x1E7E, Theme::P3Grief, 4000);
        let washes = events
            .iter()
            .filter(|(_, e)| e.sound == SoundId::RiverWash)
            .count();
        assert!(washes > 20, "P3 barely used the river ({washes} washes)");
        // Hollow: only the bottom three notes of the scale.
        for (_, e) in &events {
            if SoundId::TONAL_PALETTE.contains(&e.sound) {
                assert!(
                    TONAL_LOW.contains(&e.sound),
                    "P3 emitted the high note {:?}",
                    e.sound
                );
            }
        }
        // Drone accents are thicker here than in the warm movement.
        let drones = events
            .iter()
            .filter(|(_, e)| matches!(e.sound, SoundId::DroneLayerA | SoundId::DroneLayerB))
            .count();
        let (_, p1) = stats(0x1E7E, Theme::P1FirstLight, 4000);
        let p1_drones = p1
            .iter()
            .filter(|(_, e)| matches!(e.sound, SoundId::DroneLayerA | SoundId::DroneLayerB))
            .count();
        assert!(
            drones > p1_drones,
            "P3 drones {drones} <= P1 drones {p1_drones}"
        );
    }

    #[test]
    fn p4_closing_is_near_silence_plus_a_heartbeat() {
        let (beats, events) = stats(0xC105, Theme::P4Closing, 4000);
        let hearts = events
            .iter()
            .filter(|(_, e)| matches!(e.sound, SoundId::HeartLub | SoundId::HeartDub))
            .count();
        let other = events.len() - hearts;
        // The heartbeat is nearly all there is.
        assert!(hearts > other * 4, "P4: hearts={hearts}, other={other}");
        let other_rate = other as f64 / beats as f64;
        assert!(
            other_rate < 0.08,
            "P4 non-heartbeat rate {other_rate} too busy"
        );

        // lub on the downbeat, dub on the beat right after it — never elsewhere.
        for (t, e) in &events {
            match e.sound {
                SoundId::HeartLub => assert_eq!(G.classify(*t), Some(BeatClass::Strong)),
                SoundId::HeartDub => {
                    assert_eq!(G.classify(*t), Some(BeatClass::Weak));
                    assert_eq!(
                        beat_in_bar(*t, G),
                        1,
                        "heart_dub fired on beat {} of the bar",
                        beat_in_bar(*t, G)
                    );
                }
                _ => {}
            }
        }
        // The pair actually pairs up: most bars with a lub get their dub.
        let lubs = events
            .iter()
            .filter(|(_, e)| e.sound == SoundId::HeartLub)
            .count();
        let dubs = events
            .iter()
            .filter(|(_, e)| e.sound == SoundId::HeartDub)
            .count();
        assert!(
            lubs > 40 && dubs > 20,
            "P4 heartbeat too rare (lub={lubs}, dub={dubs})"
        );
        assert!(
            dubs <= lubs,
            "more answers than beats (lub={lubs}, dub={dubs})"
        );
    }

    #[test]
    fn movements_are_reproducible_and_seed_sensitive() {
        for theme in Theme::MOVEMENTS {
            let a = run(0x2468, theme, 600);
            let b = run(0x2468, theme, 600);
            assert_eq!(a, b, "{theme:?} not reproducible");
            let c = run(0x1357, theme, 600);
            assert_ne!(a, c, "{theme:?} ignored its seed");
        }
    }

    #[test]
    fn movement_densities_follow_the_designed_curve() {
        // A movement is characterised by how much *non-motif* material it plays:
        // P1 sparse-and-tonal, P2 thin under its pulse, P3 fuller, P4 nearly
        // nothing. Compare only the generic layer so a motif's steadiness does
        // not mask the curve.
        let generic = |theme: Theme| -> f64 {
            let cfg = DirectorConfig::for_theme(theme);
            let (beats, events) = stats(0x9001, theme, 6000);
            let n = events
                .iter()
                .filter(|(_, e)| {
                    cfg.strong_motif != Some(e.sound) && cfg.weak_motif != Some(e.sound)
                })
                .count();
            n as f64 / beats as f64
        };
        let p1 = generic(Theme::P1FirstLight);
        let p3 = generic(Theme::P3Grief);
        let p4 = generic(Theme::P4Closing);
        assert!(p4 < p1, "P4 ({p4}) must be quieter than P1 ({p1})");
        assert!(p1 < p3, "P1 ({p1}) must be sparser than P3 ({p3})");
        assert!(p4 < 0.08, "P4 generic layer {p4} is not near-silence");
    }

    // ─────────────────────────── solve silence ───────────────────────────────

    #[test]
    fn silence_window_suppresses_theme_events_then_expires() {
        let mut d = Director::new(Theme::P2TheGaze, 0x5011);
        assert!(!d.is_silent());

        // Hold the breath for 4 s, then run beats while advancing time in 0.5 s
        // steps: nothing may be emitted until the window has run out.
        d.hold_breath(4.0);
        assert!(d.is_silent());
        assert_eq!(d.silence_remaining(), 4.0);

        let mut during = 0usize;
        for step in 0..8 {
            for t in (step * 8)..(step * 8 + 8) {
                if d.on_beat(t, G).is_some() {
                    during += 1;
                }
            }
            d.advance_silence(0.5);
        }
        assert_eq!(during, 0, "theme audio leaked through the silence window");
        assert!(!d.is_silent(), "window did not expire");
        assert_eq!(d.silence_remaining(), 0.0);

        // And afterwards the theme comes back.
        let mut after = 0usize;
        for t in 64..320u64 {
            if d.on_beat(t, G).is_some() {
                after += 1;
            }
        }
        assert!(after > 0, "theme never resumed after the silence window");
    }

    #[test]
    fn silence_does_not_shift_the_stream_that_follows_it() {
        // The window removes events; it must not re-phase the arrangement. A
        // muted director and a free-running one agree on every beat outside the
        // window.
        let mut muted = Director::new(Theme::P3Grief, 0x7AB0);
        let mut free = Director::new(Theme::P3Grief, 0x7AB0);
        muted.hold_breath(2.0);

        let mut inside = 0usize;
        for t in 0..400u64 {
            let a = muted.on_beat(t, G);
            let b = free.on_beat(t, G);
            if t < 100 {
                // Still muted (time is advanced below only after tick 100).
                assert!(a.is_none(), "event {a:?} escaped the window at tick {t}");
                if b.is_some() {
                    inside += 1;
                }
            } else {
                assert_eq!(a, b, "stream diverged after the window at tick {t}");
            }
            if t == 99 {
                muted.advance_silence(2.0);
                assert!(!muted.is_silent());
            }
        }
        assert!(inside > 0, "the test never actually suppressed anything");
    }

    #[test]
    fn reentrant_hold_extends_rather_than_stacks() {
        let mut d = Director::new(Theme::P4Closing, 0x9);
        d.hold_breath(4.0);
        d.advance_silence(1.5); // 2.5 s left
        assert_eq!(d.silence_remaining(), 2.5);

        // A second solve while the first window is open: extend to the longer of
        // the two, never 2.5 + 4.0.
        d.hold_breath(4.0);
        assert_eq!(d.silence_remaining(), 4.0);

        // A *shorter* request must not cut the open window short.
        d.hold_breath(1.0);
        assert_eq!(d.silence_remaining(), 4.0);

        // Total elapsed silence from the first call is bounded by the longest
        // single request, not the sum.
        d.advance_silence(4.0);
        assert!(!d.is_silent());
    }

    #[test]
    fn hold_breath_clamps_and_ignores_nonsense() {
        let mut d = Director::new(Theme::Default, 0x11);
        d.hold_breath(0.0);
        assert!(!d.is_silent(), "a zero-length hold must be a no-op");
        d.hold_breath(-3.0);
        assert!(!d.is_silent(), "a negative hold must be a no-op");
        d.hold_breath(f32::NAN);
        assert!(!d.is_silent(), "NaN must be a no-op");

        d.hold_breath(1_000.0);
        assert_eq!(
            d.silence_remaining(),
            Director::MAX_SILENCE_SECS,
            "an absurd request must be clamped"
        );

        // Advancing with nonsense dt cannot resurrect or corrupt the window.
        d.advance_silence(f32::NAN);
        d.advance_silence(-5.0);
        assert_eq!(d.silence_remaining(), Director::MAX_SILENCE_SECS);
        d.advance_silence(f32::INFINITY);
        assert_eq!(d.silence_remaining(), Director::MAX_SILENCE_SECS);

        d.release_breath();
        assert!(!d.is_silent());
        // The designed window sits inside the 3–5 s the progression asks for.
        assert!((3.0..=5.0).contains(&Director::SOLVE_SILENCE_SECS));
    }

    #[test]
    fn silence_survives_a_theme_swap() {
        // A level transition during the held breath must not cancel it (the
        // ambience keeps running either way — only the theme layer is muted).
        let mut d = Director::new(Theme::P3Grief, 0x3);
        d.hold_breath(3.0);
        d.set_theme(Theme::P4Closing);
        assert!(d.is_silent());
        assert_eq!(d.theme(), Theme::P4Closing);
        for t in 0..32u64 {
            assert!(d.on_beat(t, G).is_none());
        }
    }

    #[test]
    fn theme_swap_keeps_stream_deterministic() {
        // Two directors that take the same swap path produce the same stream.
        let mut a = Director::new(Theme::Cold, 0x3333);
        let mut b = Director::new(Theme::Cold, 0x3333);
        let mut va = Vec::new();
        let mut vb = Vec::new();
        for t in 0..200 {
            if t == 100 {
                a.set_theme(Theme::Garden);
                b.set_theme(Theme::Garden);
            }
            va.push(a.on_beat(t, G));
            vb.push(b.on_beat(t, G));
        }
        assert_eq!(va, vb);
    }
}
