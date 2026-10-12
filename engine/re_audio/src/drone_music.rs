//! `drone_music` — the evolving background MUSIC layer. **Pure compute, no device.**
//!
//! [`crate::soundscape::render_bed`] is the room's *tone* — a short (12 s),
//! fixed-root loop that establishes a mood and is meant to be heard as part
//! of the space. This module is a second, slower layer sitting further under
//! everything: a long-form harmonic drone that never repeats audibly across a
//! multi-minute play session and occasionally shifts its harmonic centre, the
//! way an ambient score moves without ever resolving into a "song".
//!
//! It deliberately reuses the mood taxonomy rather than inventing a second
//! one: [`chord_degrees`] is keyed by the same four [`AmbienceMood`]s the
//! rest of the audio layer already uses (see `ambience`/`soundscape`), and
//! [`render_music_segment`]'s register sits an octave below
//! [`crate::soundscape::harmony_root`] so the two layers never crowd the same
//! band.
//!
//! * [`render_music_segment`] — one harmonic segment: a sub layer (root, one
//!   octave down, slow amplitude swell), 4 detuned partials run through a
//!   slowly LFO-swept low-pass (the "filter movement"), and an airy
//!   band-passed noise layer. Seamless-looped like the rest of the ambience
//!   ([`crate::dsp::make_seamless`]).
//! * [`chord_degrees`] — a small, harmonically related set of semitone shifts
//!   per mood, so a shift always reads as a deliberate modulation rather than
//!   a random wander.
//! * [`MusicSequencer`] — pure, seeded, and a function of elapsed time only
//!   (not of call count): [`MusicSequencer::state_at`] answers "which chord is
//!   current, which is next, and how far through the crossfade are we" for
//!   any elapsed time, replayably. Chord holds are long (minutes, jittered)
//!   and crossfades are long (seconds), so the shift itself is inaudible as
//!   an edit — it reads as the room's mood drifting.
//!
//! Startup cost: a segment is [`MUSIC_SEG_SECS`] long — long enough that two
//! full plays never share a moment, short enough that rendering one costs
//! about the same as one [`crate::soundscape::SceneAssets`] layer (same
//! per-sample cost, same order of magnitude of samples). The device layer
//! never renders more than the two segments the sequencer is currently
//! between ("current" and "next"); a segment further out is rendered on a
//! worker thread only once the sequencer commits to it (mirrors
//! `crate::playback::Player`'s existing background scene render — see
//! `playback.rs`'s `request_scene`/`scene_pending` for the pattern this
//! follows). Nothing here blocks `SynthBank::build`.

use crate::acoustics::{Room, RoomVerb};
use crate::ambience::AmbienceMood;
use crate::dsp::{make_seamless, secs, Drift, OnePole, Stereo};
use crate::soundscape::harmony_root;
use crate::synth::{noise, osc, Lcg, Wave, SAMPLE_RATE};

const DT: f32 = 1.0 / SAMPLE_RATE as f32;

/// Length of one rendered chord segment (seconds). Long enough that its own
/// loop point is never heard in ordinary play; the [`MusicSequencer`] moves
/// on to a new chord (via a long crossfade) well before a segment's own loop
/// would repeat often enough to notice.
pub const MUSIC_SEG_SECS: f32 = 22.0;
/// Loop-seam crossfade length within one rendered segment.
const MUSIC_SEAM_SECS: f32 = 3.0;

/// Harmonic partial ratios above the sub (root × 1, 2, 3, 4 — an open
/// harmonic stack, not a specific chord voicing; the chord IS the root shift).
const PARTIAL_RATIOS: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

/// Sub-layer level (root, one octave down).
pub const MUSIC_SUB_MIX: f32 = 0.55;
/// Detuned-partials layer level (post slow-filter).
pub const MUSIC_PARTIAL_MIX: f32 = 0.65;
/// Airy noise layer level.
pub const MUSIC_AIR_MIX: f32 = 0.22;
/// Slow filter sweep: minimum cutoff (Hz) — the partial stack fully muffled.
pub const MUSIC_FILTER_MIN_HZ: f32 = 550.0;
/// Slow filter sweep: how far above the minimum the sweep opens (Hz).
pub const MUSIC_FILTER_SWEEP_HZ: f32 = 2600.0;
/// Slow filter sweep period (seconds) — one open→closed cycle.
pub const MUSIC_FILTER_LFO_PERIOD_S: f32 = 9.0;
/// Overall music level relative to the ambience bed (mixed well under SFX).
pub const MUSIC_LEVEL: f32 = 0.42;

/// A small, harmonically-related set of root shifts (semitones) per mood.
/// Picking among these always sounds like an intentional modulation — a
/// subdominant, a relative minor, a raised sixth — never an arbitrary detune.
/// Index 0 is always the mood's home root (shift 0).
pub fn chord_degrees(mood: AmbienceMood) -> &'static [f32] {
    match mood {
        AmbienceMood::ColdVoid => &[0.0, -5.0, 3.0, -2.0],
        AmbienceMood::ServerHum => &[0.0, 5.0, -3.0, 2.0],
        AmbienceMood::OrganicDecay => &[0.0, -4.0, 3.0, -7.0],
        // A raised sixth (tritone-ish, 6 semitones) belongs in the tense
        // mood only — the one place an unresolved shift is the point.
        AmbienceMood::AlarmTension => &[0.0, 1.0, -1.0, 6.0],
    }
}

/// Render `n + xfade` samples with `gen`, then fold them into an `n`-sample
/// seamless loop (mirrors `soundscape::seamless`; duplicated rather than
/// shared because it is a two-line wrapper and the two modules stay
/// independent of each other's internals).
fn seamless(n: usize, xfade: usize, mut gen: impl FnMut() -> f32) -> Vec<f32> {
    let raw: Vec<f32> = (0..n + xfade).map(|_| gen()).collect();
    make_seamless(&raw, n, xfade)
}

/// Render one chord segment (dry, stereo, seamless): sub + detuned partials
/// through a slow filter sweep + airy noise, all at `mood`'s root shifted by
/// `semitone_shift`. The register sits an octave below
/// [`harmony_root`], so the drone music never crowds the mid-register
/// harmony pad.
pub fn render_music_segment(mood: AmbienceMood, semitone_shift: f32, seed: u32) -> Stereo {
    let root = 0.5 * harmony_root(mood) * 2f32.powf(semitone_shift / 12.0);
    let n = secs(MUSIC_SEG_SECS);
    let xf = secs(MUSIC_SEAM_SECS.min(MUSIC_SEG_SECS * 0.25));
    let partial_weight_sum: f32 = (0..PARTIAL_RATIOS.len())
        .map(|k| 1.0 / (k as f32 + 1.0))
        .sum();

    let ch = |side: f32, ch_seed: u32| -> Vec<f32> {
        let mut phase = [0.0f32; PARTIAL_RATIOS.len()];
        let mut sub_phase = 0.0f32;
        let mut filt_lfo = Drift::new(ch_seed ^ 0xF17E_0001, MUSIC_FILTER_LFO_PERIOD_S);
        let mut lp = OnePole::lowpass(MUSIC_FILTER_MIN_HZ);
        let mut air_lcg = Lcg(ch_seed ^ 0xA112_0002);
        let mut air_hp = OnePole::lowpass(3000.0);
        let mut air_lp = OnePole::lowpass(9500.0);
        let mut air_lp2 = OnePole::lowpass(9500.0);
        let mut air_swell = Drift::new(ch_seed ^ 0xA113_0003, 11.0);
        seamless(n, xf, move || {
            // Sub layer: a slow, near-inaudible-detune sine an octave below root.
            sub_phase = (sub_phase + root * 0.5 * (1.0 + side * 0.0008) * DT).fract();
            let sub = osc(Wave::Sine, sub_phase);

            // Detuned harmonic stack, each partial quieter and wider-detuned
            // the higher it sits (same shape as soundscape::render_bed).
            let mut s = 0.0f32;
            for (k, &r) in PARTIAL_RATIOS.iter().enumerate() {
                let cents = side * (2.0 + 1.7 * k as f32);
                let f = root * r * 2f32.powf(cents / 1200.0);
                phase[k] = (phase[k] + f * DT).fract();
                let wave = if k % 2 == 0 { Wave::Sine } else { Wave::Tri };
                s += osc(wave, phase[k]) / (k as f32 + 1.0);
            }
            s /= partial_weight_sum;

            // Slow filter movement over the partial stack: a full open→closed
            // sweep every `MUSIC_FILTER_LFO_PERIOD_S`-ish seconds (per-channel
            // drift, so L and R sweep independently — width, not a mono wah).
            lp.retune(MUSIC_FILTER_MIN_HZ + MUSIC_FILTER_SWEEP_HZ * filt_lfo.tick());
            let filtered = lp.lp(s);

            // Airy noise: band-passed, slowly swelling.
            let band = air_lp2.lp(air_lp.lp(air_hp.hp(noise(&mut air_lcg)))) * 3.0;
            let air = band * (0.15 + 0.35 * air_swell.tick());

            sub * MUSIC_SUB_MIX + filtered * MUSIC_PARTIAL_MIX + air * MUSIC_AIR_MIX
        })
    };
    let mut out = Stereo {
        l: ch(-1.0, seed.wrapping_mul(11).wrapping_add(3)),
        r: ch(1.0, seed.wrapping_mul(13).wrapping_add(5)),
    };
    out.normalize_to(0.6);
    out
}

/// Render a segment and bake it through the level's [`Room`] (same treatment
/// as [`crate::soundscape::SceneAssets`]'s bed/texture/harmony layers), so the
/// music sits "in this room" exactly as everything else does.
pub fn render_music_segment_baked(
    mood: AmbienceMood,
    semitone_shift: f32,
    room: &Room,
    seed: u32,
) -> Stereo {
    RoomVerb::new(room).bake_loop(&render_music_segment(mood, semitone_shift, seed))
}

// ───────────────────────────── the chord sequencer ───────────────────────────

/// Mean hold duration for one chord before the sequencer considers a shift
/// (seconds) — long-timescale, "minutes" not seconds.
pub const CHORD_HOLD_MEAN_S: f32 = 75.0;
/// ± jitter applied to the mean hold (seconds), so holds are irregular.
pub const CHORD_HOLD_JITTER_S: f32 = 35.0;
/// Floor on a jittered hold, so an unlucky draw can never make two chords
/// crossfade into each other with no settled time between.
const CHORD_HOLD_MIN_S: f32 = 20.0;
/// The crossfade between chords (seconds) — long enough that the shift never
/// reads as a cut or an edit, only as the room's mood drifting.
pub const CHORD_CROSSFADE_S: f32 = 8.0;

/// Which chord is active/incoming and how far through the crossfade the
/// sequencer is, at some elapsed time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MusicState {
    /// Index into [`chord_degrees`] for the mood.
    pub current: usize,
    /// Index into [`chord_degrees`] for the mood — the chord being
    /// crossfaded to. Equal to `current` while `mix == 0.0`.
    pub next: usize,
    /// Crossfade progress in `[0, 1]`: 0 = fully `current`, 1 = fully `next`.
    pub mix: f32,
}

/// A pure, seeded, elapsed-time-driven chord schedule for one mood.
///
/// [`MusicSequencer::state_at`] is a genuine function of `elapsed_secs` alone
/// (not of how many times it has been called, and not of a mutable cursor):
/// querying the same sequencer at the same time twice always returns the same
/// answer, and querying it out of order (e.g. a save/restore that jumps
/// elapsed time backward) still lands on the schedule's true segment rather
/// than drifting. The device layer stores only the mood, the seed, and a
/// start [`std::time::Instant`]; the schedule itself needs no persisted
/// state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MusicSequencer {
    mood: AmbienceMood,
    seed: u32,
}

impl MusicSequencer {
    /// A sequencer for `mood`, seeded by `seed` (typically the level/room
    /// seed, so two different rooms never share a chord schedule).
    pub fn new(mood: AmbienceMood, seed: u32) -> MusicSequencer {
        MusicSequencer { mood, seed }
    }

    /// The mood this sequencer schedules chords for.
    pub fn mood(self) -> AmbienceMood {
        self.mood
    }

    fn root_lcg(self) -> Lcg {
        Lcg(self.seed ^ 0x5EA5_0000 ^ (self.mood as u32).wrapping_mul(0x9E37_79B1))
    }

    /// The schedule at `elapsed_secs`. Walks the seeded chord sequence from
    /// t = 0, accumulating each hold's *exact* jittered length (rather than
    /// dividing elapsed time by a fixed period), so two different queries
    /// never disagree about where a segment boundary falls.
    pub fn state_at(self, elapsed_secs: f32) -> MusicState {
        let degrees = chord_degrees(self.mood);
        let len = degrees.len();
        if len <= 1 {
            return MusicState {
                current: 0,
                next: 0,
                mix: 0.0,
            };
        }
        let mut lcg = self.root_lcg();
        let mut t = 0.0f32;
        let mut current = 0usize;
        loop {
            let hold =
                (CHORD_HOLD_MEAN_S + CHORD_HOLD_JITTER_S * noise(&mut lcg)).max(CHORD_HOLD_MIN_S);
            let seg_end = t + hold;
            // Peek the next chord without consuming `lcg` for real yet, so a
            // repeated query at the same `elapsed_secs` reproduces exactly.
            let mut peek = lcg;
            let next = pick_next(&mut peek, len, current);
            if elapsed_secs.max(0.0) < seg_end {
                let fade_start = seg_end - CHORD_CROSSFADE_S;
                let mix = if elapsed_secs <= fade_start {
                    0.0
                } else {
                    ((elapsed_secs - fade_start) / CHORD_CROSSFADE_S).clamp(0.0, 1.0)
                };
                return MusicState { current, next, mix };
            }
            t = seg_end;
            current = pick_next(&mut lcg, len, current);
        }
    }
}

/// Pick a chord index different from `avoid` (so a shift is always audible),
/// deterministically from `lcg`'s next draw.
fn pick_next(lcg: &mut Lcg, len: usize, avoid: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let u = noise(lcg) * 0.5 + 0.5; // [0, 1)
    let mut idx = (u * len as f32) as usize;
    if idx >= len {
        idx = len - 1;
    }
    if idx == avoid {
        idx = (idx + 1) % len;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_are_wide_stereo_audible_and_bounded() {
        for mood in AmbienceMood::ALL {
            for &deg in chord_degrees(mood) {
                let s = render_music_segment(mood, deg, 5);
                assert_eq!(s.l.len(), secs(MUSIC_SEG_SECS));
                assert!(s.rms() > 0.02, "{mood:?} deg {deg} is nearly silent");
                assert!(s.peak() <= 1.0);
                assert_ne!(s.l, s.r, "{mood:?} deg {deg} collapsed to mono");
            }
        }
    }

    #[test]
    fn segments_loop_without_a_seam_jump() {
        let s = render_music_segment(AmbienceMood::ColdVoid, 0.0, 3);
        let n = s.l.len();
        let seam = (s.l[n - 1] - s.l[0]).abs();
        let max_step =
            s.l.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0f32, f32::max);
        assert!(
            seam <= max_step * 1.05,
            "seam {seam} vs max step {max_step}"
        );
    }

    #[test]
    fn rendering_is_deterministic() {
        let a = render_music_segment(AmbienceMood::ServerHum, 5.0, 9);
        let b = render_music_segment(AmbienceMood::ServerHum, 5.0, 9);
        assert_eq!(a, b);
        let c = render_music_segment(AmbienceMood::ServerHum, 5.0, 10);
        assert_ne!(a, c, "different seed must render different PCM");
    }

    #[test]
    fn different_chords_of_the_same_mood_sound_different() {
        let degrees = chord_degrees(AmbienceMood::OrganicDecay);
        let root = render_music_segment(AmbienceMood::OrganicDecay, degrees[0], 1);
        for &d in &degrees[1..] {
            let other = render_music_segment(AmbienceMood::OrganicDecay, d, 1);
            assert_ne!(root, other, "chord {d} sounds identical to the root");
        }
    }

    #[test]
    fn every_mood_has_its_home_root_at_index_zero() {
        for mood in AmbienceMood::ALL {
            assert_eq!(
                chord_degrees(mood)[0],
                0.0,
                "{mood:?}'s index 0 must be the home root"
            );
        }
    }

    #[test]
    fn sequencer_holds_a_chord_then_crossfades_and_moves_on() {
        let seq = MusicSequencer::new(AmbienceMood::ColdVoid, 42);
        let start = seq.state_at(0.0);
        assert_eq!(start.mix, 0.0);
        assert_eq!(start.current, 0, "starts on the home root");

        // Somewhere in the middle of a long hold: still settled, no fade.
        let mid = seq.state_at(30.0);
        assert_eq!(mid.mix, 0.0);
        assert_eq!(mid.current, start.current);

        // Walk forward until we observe a crossfade in progress, then watch
        // it complete into the announced `next`.
        let mut t = 0.0f32;
        let mut saw_fade = false;
        let mut committed = None;
        while t < 3600.0 && committed.is_none() {
            let s = seq.state_at(t);
            if s.mix > 0.0 && s.mix < 1.0 {
                saw_fade = true;
                assert_ne!(
                    s.current, s.next,
                    "a fading state must name two different chords"
                );
            }
            if saw_fade && s.mix == 0.0 && s.current != start.current {
                committed = Some(s.current);
            }
            t += 0.5;
        }
        assert!(
            saw_fade,
            "never observed an in-progress crossfade within an hour"
        );
        assert!(
            committed.is_some(),
            "never committed to a new chord within an hour"
        );
    }

    #[test]
    fn sequencer_is_a_pure_function_of_elapsed_time() {
        let seq = MusicSequencer::new(AmbienceMood::AlarmTension, 7);
        for t in [0.0f32, 1.0, 59.9, 60.0, 200.3, 900.0, 1800.0] {
            assert_eq!(
                seq.state_at(t),
                seq.state_at(t),
                "not deterministic at t={t}"
            );
        }
        // Different seeds diverge (not literally the same schedule).
        let other = MusicSequencer::new(AmbienceMood::AlarmTension, 8);
        let a: Vec<_> = (0..600)
            .map(|s| seq.state_at(s as f32 * 10.0).current)
            .collect();
        let b: Vec<_> = (0..600)
            .map(|s| other.state_at(s as f32 * 10.0).current)
            .collect();
        assert_ne!(
            a, b,
            "two different seeds produced an identical chord schedule"
        );
    }

    #[test]
    fn crossfade_never_exceeds_the_configured_window() {
        let seq = MusicSequencer::new(AmbienceMood::ServerHum, 3);
        let mut t = 0.0f32;
        let mut in_fade_samples = 0u32;
        while t < 1200.0 {
            let s = seq.state_at(t);
            if s.mix > 0.0 && s.mix < 1.0 {
                in_fade_samples += 1;
            }
            t += 0.25;
        }
        // Roughly CHORD_CROSSFADE_S worth of quarter-second samples per
        // observed fade; generously bounded (a handful of holds occur in
        // 1200s at the configured mean).
        let max_fades_expected = (1200.0 / (CHORD_HOLD_MEAN_S - CHORD_HOLD_JITTER_S)) as u32 + 2;
        let max_samples = max_fades_expected * ((CHORD_CROSSFADE_S / 0.25) as u32 + 2);
        assert!(
            in_fade_samples <= max_samples,
            "{in_fade_samples} vs bound {max_samples}"
        );
    }
}
