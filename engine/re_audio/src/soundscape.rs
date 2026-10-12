//! `soundscape` — the layered, stereo, room-baked ambience. **Pure compute, no device.**
//!
//! The old ambience was one mono drone loop per mood, upmixed to identical L/R,
//! with a fade-in/fade-out envelope that dipped every 14–16 s at the loop seam
//! and a fixed reverb that knew nothing about the room. It sounded like a tone
//! played *at* the player. This module builds the ambience *around* the player
//! from three layers plus positional sources:
//!
//! 1. **Drone bed** ([`render_bed`]) — the mood's partials rendered separately
//!    per ear: every partial detuned a few cents in opposite directions on L and
//!    R, each with its own slow amplitude drift, one partial slowly panning, and
//!    an independent noise floor per channel. The channels are *decorrelated*,
//!    which the ear hears as width. Seamless loop (equal-power tail crossfade —
//!    no periodic dip).
//! 2. **Mid texture** ([`render_texture`]) — the room's air: band-limited noise
//!    whose band and motion depend on the mood (wind, server fans, a wet churn,
//!    an alarm tremolo), again independent per ear.
//! 3. **Harmony voices** ([`render_harmony_voice`] × 6) — sustained pad tones on
//!    the mood's root: root, fifth, octave, major tenth, twelfth, and one
//!    *tension* voice a minor second above the root. Their gains are driven live
//!    by [`harmony_gains`] from the puzzle state, so **the player hears progress
//!    in the room's chord**: an unsolved room is a bare root rubbing against a
//!    minor second; each lit node fills in consonance; a refusal spikes the
//!    dissonance for a moment.
//! 4. **Sparse detail events** ([`DetailScheduler`], [`render_detail`]) — relay
//!    clicks, drips, creaks, data chirps, distant thumps; seeded, never on a
//!    fixed period, placed across the stereo field at varying distance.
//!
//! All of layers 1–4 are baked through the level's [`Room`]
//! ([`crate::acoustics::RoomVerb`]) at level load, so the early reflections and
//! the tail are those of *this* room. Positional loops ([`render_loop`]) are
//! rendered dry: they are spatialized live and the device's room reverb carries
//! them.
//!
//! Everything is seeded: the same `(mood, room, seed)` renders the same samples.

use crate::acoustics::{Room, RoomVerb};
use crate::ambience::AmbienceMood;
use crate::dsp::{make_seamless, pan_gains, secs, Drift, OnePole, Stereo};
use crate::spatial::LoopKind;
use crate::synth::{noise, osc, Lcg, Wave, SAMPLE_RATE};

/// Loop-seam crossfade length for the long layers (seconds).
const SEAM_SECS: f32 = 2.0;
/// Harmony voice loop length (seconds).
pub const HARMONY_SECS: f32 = 8.0;
/// Positional loop length (seconds).
pub const LOOP_SECS: f32 = 4.0;
/// Number of harmony voices (root, fifth, octave, tenth, twelfth, tension).
pub const HARMONY_VOICES: usize = 6;
/// Index of the dissonant tension voice in the harmony arrays.
pub const TENSION_VOICE: usize = 5;
/// Frequency ratios of the harmony voices over the mood's harmony root.
pub const HARMONY_RATIOS: [f32; HARMONY_VOICES] = [1.0, 1.5, 2.0, 2.5, 3.0, 16.0 / 15.0];

const DT: f32 = 1.0 / SAMPLE_RATE as f32;

// ── mix levels (shared by the device layer and the offline mixdown) ──
/// Mid-texture level relative to the bed.
pub const TEXTURE_LEVEL: f32 = 0.7;
/// Harmony-voice level (each voice's [`harmony_gains`] is scaled by this).
pub const HARMONY_LEVEL: f32 = 0.45;
/// Detail-event level.
pub const DETAIL_LEVEL: f32 = 0.6;
/// Positional-loop level (before spatial gain).
pub const POSITIONAL_LEVEL: f32 = 0.6;
/// Voice budget for positional loops.
pub const MAX_POSITIONAL_VOICES: usize = 6;

/// The partial voicing of a mood's drone bed: `(Hz, gain, wave)`.
fn bed_voicing(mood: AmbienceMood) -> (&'static [(f32, f32, Wave)], f32, f32) {
    // (partials, noise low-pass Hz, noise gain)
    match mood {
        AmbienceMood::ColdVoid => (
            &[
                (41.20, 1.00, Wave::Sine),
                (61.74, 0.45, Wave::Sine),
                (82.41, 0.30, Wave::Tri),
                (123.47, 0.12, Wave::Sine),
            ],
            180.0,
            0.30,
        ),
        AmbienceMood::ServerHum => (
            &[
                (50.0, 1.00, Wave::Sine),
                (100.0, 0.16, Wave::Square),
                (150.0, 0.16, Wave::Sine),
                (200.0, 0.07, Wave::Sine),
            ],
            420.0,
            0.18,
        ),
        AmbienceMood::OrganicDecay => (
            &[
                (55.0, 1.00, Wave::Sine),
                (65.41, 0.50, Wave::Tri),
                (82.41, 0.40, Wave::Sine),
                (110.0, 0.20, Wave::Sine),
            ],
            380.0,
            0.24,
        ),
        AmbienceMood::AlarmTension => (
            &[
                (110.0, 1.00, Wave::Sine),
                (116.54, 0.65, Wave::Sine),
                (164.81, 0.15, Wave::Sine),
                (220.0, 0.25, Wave::Tri),
            ],
            1100.0,
            0.10,
        ),
    }
}

/// The root the harmony voices are tuned to, per mood (mid register, a few
/// octaves above the drone root so it sits between bed and texture).
pub fn harmony_root(mood: AmbienceMood) -> f32 {
    match mood {
        AmbienceMood::ColdVoid => 164.81,     // E3 over the E1 drone
        AmbienceMood::ServerHum => 196.00,    // G3 over the ~G1 mains hum
        AmbienceMood::OrganicDecay => 220.00, // A3 over the A1 drone
        AmbienceMood::AlarmTension => 220.00, // A3 over the A2 cluster
    }
}

/// Render `n + xfade` samples with `gen`, then fold them into an `n`-sample
/// seamless loop.
fn seamless(n: usize, xfade: usize, mut gen: impl FnMut() -> f32) -> Vec<f32> {
    let raw: Vec<f32> = (0..n + xfade).map(|_| gen()).collect();
    make_seamless(&raw, n, xfade)
}

/// Layer 1: the decorrelated stereo drone bed for `mood` (dry, seamless).
pub fn render_bed(mood: AmbienceMood, secs_len: f32, seed: u32) -> Stereo {
    let (partials, noise_hz, noise_gain) = bed_voicing(mood);
    let n = secs(secs_len);
    let xf = secs(SEAM_SECS.min(secs_len * 0.25));
    let gsum: f32 = partials.iter().map(|p| p.1).sum();

    let ch = |side: f32, ch_seed: u32| -> Vec<f32> {
        let mut phase: Vec<f32> = partials
            .iter()
            .enumerate()
            .map(|(k, _)| {
                ((ch_seed
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(k as u32 * 97))
                    % 1000) as f32
                    / 1000.0
            })
            .collect();
        let mut drift: Vec<Drift> = (0..partials.len())
            .map(|k| Drift::new(ch_seed ^ (0x9E37 * (k as u32 + 1)), 3.0 + k as f32 * 1.3))
            .collect();
        let mut lcg = Lcg(ch_seed ^ 0x00B3_D0A5);
        let mut nlp = OnePole::lowpass(noise_hz);
        let mut nlp2 = OnePole::lowpass(noise_hz);
        let mut t = 0.0f32;
        seamless(n, xf, move || {
            let mut s = 0.0f32;
            for (k, &(f, g, wave)) in partials.iter().enumerate() {
                // Opposite few-cent detune per ear, growing with the partial.
                let cents = side * (3.0 + 2.5 * k as f32);
                let fk = f * 2f32.powf(cents / 1200.0);
                phase[k] = (phase[k] + fk * DT).fract();
                let amp = 0.55 + 0.45 * drift[k].tick();
                let mut v = g * amp * osc(wave, phase[k]);
                if k == 1 {
                    // One partial slowly travels across the field (~48 s cycle).
                    let pan = 0.6 * (2.0 * std::f32::consts::PI * 0.021 * t).sin();
                    let (gl, gr) = pan_gains(pan);
                    v *= if side < 0.0 { gl } else { gr } * std::f32::consts::SQRT_2;
                }
                s += v;
            }
            s /= gsum;
            let nz = nlp2.lp(nlp.lp(noise(&mut lcg))) * 6.0;
            t += DT;
            (s * (1.0 - noise_gain) + nz * noise_gain) * 0.55
        })
    };
    let l = ch(-1.0, seed.wrapping_mul(3).wrapping_add(1));
    let r = ch(1.0, seed.wrapping_mul(7).wrapping_add(2));
    Stereo { l, r }
}

/// Layer 2: the room's air — band-limited, mood-coloured noise texture
/// (dry, seamless, independent per ear).
pub fn render_texture(mood: AmbienceMood, secs_len: f32, seed: u32) -> Stereo {
    // (band low Hz, band high Hz, swell period s, swell depth, level)
    let (lo, hi, period, depth, level) = match mood {
        AmbienceMood::ColdVoid => (140.0, 700.0, 3.5, 0.85, 0.55),
        AmbienceMood::ServerHum => (900.0, 3200.0, 2.0, 0.35, 0.30),
        AmbienceMood::OrganicDecay => (220.0, 950.0, 1.2, 0.70, 0.42),
        AmbienceMood::AlarmTension => (600.0, 2600.0, 0.9, 0.55, 0.28),
    };
    let n = secs(secs_len);
    let xf = secs(SEAM_SECS.min(secs_len * 0.25));
    let ch = |ch_seed: u32| -> Vec<f32> {
        let mut lcg = Lcg(ch_seed);
        let mut hp = OnePole::lowpass(lo);
        let mut lp1 = OnePole::lowpass(hi);
        let mut lp2 = OnePole::lowpass(hi);
        let mut swell = Drift::new(ch_seed ^ 0x51E1, period);
        let mut churn = Drift::new(ch_seed ^ 0xC4A2, 0.18);
        let mut ph = 0.0f32;
        let mut ph2 = 0.0f32;
        let mut wob = Drift::new(ch_seed ^ 0x0B0B, 2.5);
        seamless(n, xf, move || {
            let x = noise(&mut lcg);
            let band = lp2.lp(lp1.lp(hp.hp(x))) * 3.0;
            let sw = swell.tick();
            let mut amp = (1.0 - depth) + depth * sw * sw;
            let mut tone = 0.0f32;
            match mood {
                AmbienceMood::ServerHum => {
                    // A fan whine that wanders a fraction of a percent.
                    let f = 1180.0 * (1.0 + 0.004 * (wob.tick() - 0.5));
                    ph = (ph + f * DT).fract();
                    tone = 0.10 * osc(Wave::Sine, ph);
                }
                AmbienceMood::OrganicDecay => {
                    // A slow wet churn: fast random amplitude flutter.
                    amp *= 0.55 + 0.45 * churn.tick();
                }
                AmbienceMood::AlarmTension => {
                    // Two faint high tones a semitone-ish apart: a far alarm beat.
                    ph = (ph + 880.0 * DT).fract();
                    ph2 = (ph2 + 932.33 * DT).fract();
                    tone = 0.05 * (osc(Wave::Sine, ph) + osc(Wave::Sine, ph2));
                }
                AmbienceMood::ColdVoid => {}
            }
            (band * amp + tone) * level
        })
    };
    Stereo {
        l: ch(seed ^ 0x7E71),
        r: ch(seed.rotate_left(7) ^ 0x7E72),
    }
}

/// Layer 3: one sustained harmony voice at `freq` (dry, seamless, chorused
/// across the ears).
pub fn render_harmony_voice(freq: f32, secs_len: f32, seed: u32) -> Stereo {
    let n = secs(secs_len);
    let xf = secs(SEAM_SECS.min(secs_len * 0.25));
    let ch = |side: f32, ch_seed: u32| -> Vec<f32> {
        let (mut a, mut b, mut c) = (0.0f32, 0.31f32, 0.67f32);
        let mut swell = Drift::new(ch_seed, 4.5);
        let fa = freq * (1.0 + side * 0.0015);
        let fb = freq * (1.0 - side * 0.0022);
        seamless(n, xf, move || {
            a = (a + fa * DT).fract();
            b = (b + fb * DT).fract();
            c = (c + 2.0 * freq * DT).fract();
            let s = 0.5 * osc(Wave::Sine, a) + 0.4 * osc(Wave::Sine, b) + 0.1 * osc(Wave::Tri, c);
            s * (0.65 + 0.35 * swell.tick()) * 0.5
        })
    };
    Stereo {
        l: ch(-1.0, seed ^ 0x4A11),
        r: ch(1.0, seed ^ 0x4A12),
    }
}

/// The six harmony voice frequencies for a mood.
pub fn harmony_freqs(mood: AmbienceMood) -> [f32; HARMONY_VOICES] {
    let root = harmony_root(mood);
    HARMONY_RATIOS.map(|r| root * r)
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The harmony mix for a puzzle `progress` in `[0, 1]` (lit fraction of the
/// network, 1 when solved) and a transient `tension` in `[0, 1]` (refusal
/// spikes). Returns the linear gain of each voice
/// (root, fifth, octave, tenth, twelfth, tension).
///
/// Contract (tested): consonant voices only ever *enter* as progress rises;
/// the tension voice only ever *fades* as progress rises and is pushed up by
/// the `tension` spike; at `progress == 1` with no spike the chord is fully
/// consonant (tension silent).
pub fn harmony_gains(progress: f32, tension: f32) -> [f32; HARMONY_VOICES] {
    let p = progress.clamp(0.0, 1.0);
    let t = tension.clamp(0.0, 1.0);
    [
        0.55 + 0.25 * p,
        0.80 * smoothstep(0.08, 0.35, p),
        0.70 * smoothstep(0.30, 0.60, p),
        0.60 * smoothstep(0.55, 0.85, p),
        0.50 * smoothstep(0.85, 1.00, p),
        (0.45 * (1.0 - p).powf(1.5) + 0.9 * t).clamp(0.0, 1.0),
    ]
}

/// Puzzle progress as the harmony sees it: the lit fraction of the network,
/// snapped to 1 once solved.
pub fn harmony_progress(lit: u32, total: u32, solved: bool) -> f32 {
    if solved {
        return 1.0;
    }
    if total == 0 {
        return 0.0;
    }
    (lit as f32 / total as f32).clamp(0.0, 1.0)
}

/// A decaying harmonic-tension level, spiked by refusal cues.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tension {
    level: f32,
}

impl Tension {
    /// Half-life of a tension spike (seconds).
    pub const HALF_LIFE: f32 = 1.2;

    /// Raise the level to at least `amount` (spikes do not stack past 1).
    pub fn pulse(&mut self, amount: f32) {
        self.level = self.level.max(amount.clamp(0.0, 1.0));
    }

    /// Age by `dt` seconds.
    pub fn decay(&mut self, dt: f32) {
        if dt > 0.0 {
            self.level *= 0.5f32.powf(dt / Self::HALF_LIFE);
            if self.level < 1e-4 {
                self.level = 0.0;
            }
        }
    }

    /// Current level in `[0, 1]`.
    pub fn level(&self) -> f32 {
        self.level
    }
}

// ───────────────────────────── detail events ────────────────────────────────

/// The sparse ambient detail events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DetailKind {
    /// A relay latching somewhere in the walls: two dry clicks.
    RelayClick,
    /// A drop of condensation.
    Drip,
    /// Metal under slow stress.
    Creak,
    /// A burst of data: three rising chirps.
    DataChirp,
    /// Heavy machinery far away.
    DistantThump,
}

impl DetailKind {
    /// Every kind, in canonical order.
    pub const ALL: [DetailKind; 5] = [
        DetailKind::RelayClick,
        DetailKind::Drip,
        DetailKind::Creak,
        DetailKind::DataChirp,
        DetailKind::DistantThump,
    ];

    /// This kind's index into [`detail_palette`]'s weight array.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Relative weight of each detail kind per mood (index = `DetailKind`), and
/// the mean gap between events (seconds).
fn detail_palette(mood: AmbienceMood) -> ([u32; 5], f32) {
    match mood {
        AmbienceMood::ColdVoid => ([1, 3, 4, 0, 2], 7.0),
        AmbienceMood::ServerHum => ([5, 1, 1, 4, 1], 4.5),
        AmbienceMood::OrganicDecay => ([1, 6, 3, 0, 1], 5.0),
        AmbienceMood::AlarmTension => ([4, 0, 2, 3, 3], 4.0),
    }
}

/// Render one detail event (mono, dry).
pub fn render_detail(kind: DetailKind, seed: u32) -> Vec<f32> {
    let mut lcg = Lcg(seed ^ 0xDE7A);
    let mut out;
    match kind {
        DetailKind::RelayClick => {
            out = vec![0.0f32; secs(0.16)];
            for &at in &[0usize, secs(0.055)] {
                let mut ph = 0.0f32;
                for i in 0..secs(0.03) {
                    let e = (-(i as f32) / (0.004 * SAMPLE_RATE as f32)).exp();
                    ph = (ph + 2300.0 * DT).fract();
                    out[at + i] += e * (0.6 * noise(&mut lcg) + 0.5 * osc(Wave::Sine, ph));
                }
            }
        }
        DetailKind::Drip => {
            out = vec![0.0f32; secs(0.12)];
            let mut ph = 0.0f32;
            let n = out.len();
            for (i, o) in out.iter_mut().enumerate() {
                let t = i as f32 / n as f32;
                let f = 1600.0 + (700.0 - 1600.0) * t.sqrt();
                ph = (ph + f * DT).fract();
                let e = (-(i as f32) / (0.02 * SAMPLE_RATE as f32)).exp();
                *o = e * osc(Wave::Sine, ph);
            }
        }
        DetailKind::Creak => {
            out = vec![0.0f32; secs(0.95)];
            let mut ph = 0.0f32;
            let mut lp = OnePole::lowpass(900.0);
            let mut jitter = Drift::new(seed, 0.05);
            let n = out.len();
            for (i, o) in out.iter_mut().enumerate() {
                let t = i as f32 / n as f32;
                let f = (70.0 + 25.0 * t) * (1.0 + 0.06 * (jitter.tick() - 0.5));
                ph = (ph + f * DT).fract();
                let env = (t * 6.0).min(1.0) * (1.0 - t).powf(1.5);
                *o = lp.lp(osc(Wave::Saw, ph)) * env * 2.0;
            }
        }
        DetailKind::DataChirp => {
            out = vec![0.0f32; secs(0.26)];
            for (k, &f) in [2200.0f32, 2600.0, 3100.0].iter().enumerate() {
                let at = secs(0.07) * k;
                let mut ph = 0.0f32;
                let len = secs(0.04);
                for i in 0..len {
                    ph = (ph + f * DT).fract();
                    let e = (1.0 - i as f32 / len as f32).powi(2);
                    out[at + i] += e * osc(Wave::Tri, ph) * 0.7;
                }
            }
        }
        DetailKind::DistantThump => {
            out = vec![0.0f32; secs(0.5)];
            let mut ph = 0.0f32;
            let mut lp = OnePole::lowpass(160.0);
            let n = out.len();
            for (i, o) in out.iter_mut().enumerate() {
                let t = i as f32 / n as f32;
                ph = (ph + (52.0 - 14.0 * t) * DT).fract();
                let e = (-(i as f32) / (0.12 * SAMPLE_RATE as f32)).exp();
                *o = e * (osc(Wave::Sine, ph) + 0.4 * lp.lp(noise(&mut lcg)));
            }
        }
    }
    let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > 0.0 {
        for v in &mut out {
            *v *= 0.6 / peak;
        }
    }
    out
}

/// One scheduled detail event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetailEvent {
    /// Which detail sound to render.
    pub kind: DetailKind,
    /// Stereo position.
    pub pan: f32,
    /// Linear gain (a far event is quieter).
    pub gain: f32,
    /// Pitch variation.
    pub rate: f64,
}

/// Seeded scheduler for the sparse detail layer: irregular gaps around the
/// mood's mean, weighted kind choice, random placement in the field.
#[derive(Clone, Debug)]
pub struct DetailScheduler {
    lcg: Lcg,
    mood: AmbienceMood,
    next_at: f32,
}

impl DetailScheduler {
    /// A scheduler for `mood`, seeded for reproducibility.
    pub fn new(mood: AmbienceMood, seed: u32) -> DetailScheduler {
        let mut s = DetailScheduler {
            lcg: Lcg(seed ^ 0x5CED),
            mood,
            next_at: 0.0,
        };
        s.next_at = s.gap();
        s
    }

    fn unit(&mut self) -> f32 {
        noise(&mut self.lcg) * 0.5 + 0.5
    }

    fn gap(&mut self) -> f32 {
        let (_, mean) = detail_palette(self.mood);
        mean * (0.45 + 1.1 * self.unit())
    }

    /// Return the next event due at or before `now_secs` (time since the
    /// scheduler started), or `None`. Call repeatedly until `None`.
    pub fn poll(&mut self, now_secs: f32) -> Option<DetailEvent> {
        if now_secs < self.next_at {
            return None;
        }
        let (weights, _) = detail_palette(self.mood);
        let total: u32 = weights.iter().sum();
        let mut pick = (self.unit() * total as f32) as u32;
        let mut kind = DetailKind::ALL[0];
        for (k, &w) in weights.iter().enumerate() {
            if pick < w {
                kind = DetailKind::ALL[k];
                break;
            }
            pick -= w;
        }
        let pan = (self.unit() * 2.0 - 1.0) * 0.9;
        let distance = self.unit();
        let gain = 0.8 - 0.5 * distance;
        let rate = 0.9 + 0.2 * f64::from(self.unit());
        self.next_at += self.gap();
        Some(DetailEvent {
            kind,
            pan,
            gain,
            rate,
        })
    }
}

// ───────────────────────────── positional loops ─────────────────────────────

/// Render a positional loop (mono, dry, seamless, peak ≈ 0.5).
pub fn render_loop(kind: LoopKind) -> Vec<f32> {
    let n = secs(LOOP_SECS);
    let xf = secs(0.5);
    let seed = 0x100 + kind.index() as u32;
    let mut lcg = Lcg(seed);
    let mut out = match kind {
        LoopKind::LampHum => {
            let (mut a, mut b) = (0.0f32, 0.0f32);
            let mut lp = OnePole::lowpass(700.0);
            let mut flicker = Drift::new(seed, 0.2);
            let mut crackle = 0usize;
            seamless(n, xf, move || {
                a = (a + 100.0 * DT).fract();
                b = (b + 200.0 * DT).fract();
                let buzz = lp.lp(osc(Wave::Saw, a)) * 0.7 + 0.25 * osc(Wave::Sine, b);
                if crackle == 0 && noise(&mut lcg) > 0.9993 {
                    crackle = 90;
                }
                let c = if crackle > 0 {
                    crackle -= 1;
                    noise(&mut lcg) * 0.5
                } else {
                    0.0
                };
                buzz * (0.85 + 0.15 * flicker.tick()) + c
            })
        }
        LoopKind::PipeHum => {
            let (mut a, mut b) = (0.0f32, 0.0f32);
            let mut lp = OnePole::lowpass(260.0);
            let mut gurgle = Drift::new(seed, 0.25);
            seamless(n, xf, move || {
                a = (a + 72.0 * DT).fract();
                b = (b + 144.0 * DT).fract();
                let flow = lp.lp(noise(&mut lcg)) * 5.0;
                (0.5 * osc(Wave::Sine, a) + 0.2 * osc(Wave::Sine, b) + 0.6 * flow)
                    * (0.6 + 0.4 * gurgle.tick())
            })
        }
        LoopKind::NodeGlow => {
            let (mut a, mut b, mut c) = (0.0f32, 0.4f32, 0.0f32);
            let mut swell = Drift::new(seed, 1.5);
            seamless(n, xf, move || {
                a = (a + 440.0 * DT).fract();
                b = (b + 441.3 * DT).fract();
                c = (c + 880.0 * DT).fract();
                (osc(Wave::Sine, a) + 0.8 * osc(Wave::Sine, b) + 0.15 * osc(Wave::Sine, c))
                    * (0.7 + 0.3 * swell.tick())
            })
        }
        LoopKind::EmitterHiss => {
            let mut hp = OnePole::lowpass(2500.0);
            let mut lp = OnePole::lowpass(7000.0);
            let mut swell = Drift::new(seed, 0.4);
            seamless(n, xf, move || {
                lp.lp(hp.hp(noise(&mut lcg))) * (0.4 + 0.6 * swell.tick())
            })
        }
    };
    let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > 0.0 {
        for v in &mut out {
            *v *= 0.5 / peak;
        }
    }
    out
}

/// Pitch of a pipe's hum for a power `load` in `[0, 1]` (used / budget, or the
/// ON share when the level has no budget): an idle grid hums low, a grid at its
/// limit strains ~a fourth higher.
pub fn pipe_hum_rate(load: f32) -> f64 {
    0.85 + 0.30 * f64::from(load.clamp(0.0, 1.0))
}

/// Level of a pipe's hum for a power `load`: louder as the grid works harder.
pub fn pipe_hum_level(load: f32) -> f32 {
    0.45 + 0.55 * load.clamp(0.0, 1.0)
}

/// Pitch of a lit node's glow: node `index` on an A-major-pentatonic ladder
/// (the glow loop is rendered at A4), so a lit network sounds as a chord.
pub fn node_glow_rate(index: usize) -> f64 {
    const STEPS: [f64; 5] = [0.0, 2.0, 4.0, 7.0, 9.0];
    let octave = (index / STEPS.len()) as f64;
    // Start an octave down (A3) and climb.
    2f64.powf((STEPS[index % STEPS.len()] + 12.0 * octave - 12.0) / 12.0)
        .min(4.0)
}

// ───────────────────────────── the scene bundle ─────────────────────────────

/// What to render for a level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneSpec {
    /// Which ambience mood drives the drone bed and detail palette.
    pub mood: AmbienceMood,
    /// The room the reverb/early-reflections processor bakes in.
    pub room: Room,
    /// Deterministic seed for every layer this spec renders.
    pub seed: u32,
    /// Length of the bed / texture loops (seconds).
    pub bed_secs: f32,
    /// Length of each harmony voice loop (seconds).
    pub harmony_secs: f32,
}

impl SceneSpec {
    /// The shipping lengths (12 s beds, 8 s harmony voices).
    pub fn new(mood: AmbienceMood, room: Room, seed: u32) -> SceneSpec {
        SceneSpec {
            mood,
            room,
            seed,
            bed_secs: 12.0,
            harmony_secs: HARMONY_SECS,
        }
    }
}

/// Every room-baked layer of a level's ambience, ready to loop.
#[derive(Clone, Debug)]
pub struct SceneAssets {
    /// The spec these assets were rendered from.
    pub spec: SceneSpec,
    /// Layer 1 (seamless loop).
    pub bed: Stereo,
    /// Layer 2 (seamless loop).
    pub texture: Stereo,
    /// Layer 3: one seamless loop per harmony voice.
    pub harmony: Vec<Stereo>,
    /// Layer 4: one room-baked one-shot per [`DetailKind`] (index = kind).
    pub details: Vec<Stereo>,
}

impl SceneAssets {
    /// Render and room-bake every layer. Pure and deterministic; heavy enough
    /// (~a second of CPU in a debug build) that the device layer runs it on a
    /// worker thread.
    pub fn render(spec: SceneSpec) -> SceneAssets {
        let bake = |dry: &Stereo| RoomVerb::new(&spec.room).bake_loop(dry);
        let mut bed = bake(&render_bed(spec.mood, spec.bed_secs, spec.seed));
        bed.normalize_to(0.55);
        let mut texture = bake(&render_texture(spec.mood, spec.bed_secs, spec.seed ^ 0x77));
        texture.normalize_to(0.35);
        let harmony = harmony_freqs(spec.mood)
            .iter()
            .enumerate()
            .map(|(k, &f)| {
                let mut s = bake(&render_harmony_voice(
                    f,
                    spec.harmony_secs,
                    spec.seed ^ (k as u32 * 131),
                ));
                s.normalize_to(0.30);
                s
            })
            .collect();
        let details = DetailKind::ALL
            .iter()
            .map(|&k| {
                let dry = render_detail(k, spec.seed ^ (k.index() as u32 * 7919));
                let mut s = RoomVerb::new(&spec.room)
                    .process(&Stereo::from_mono(&dry), spec.room.rt60().min(2.0));
                s.normalize_to(0.5);
                s
            })
            .collect();
        SceneAssets {
            spec,
            bed,
            texture,
            harmony,
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beds_are_wide_stereo_not_upmixed_mono() {
        for mood in AmbienceMood::ALL {
            let bed = render_bed(mood, 3.0, 1);
            assert_eq!(bed.l.len(), bed.r.len());
            assert_ne!(bed.l, bed.r, "{} bed is mono", mood.name());
            let c = bed.lr_correlation();
            assert!(c < 0.97, "{} bed barely decorrelated ({c})", mood.name());
            assert!(bed.rms() > 0.02, "{} bed is silent", mood.name());
            assert!(bed.peak() <= 1.0);
        }
    }

    #[test]
    fn texture_is_decorrelated_noise() {
        for mood in AmbienceMood::ALL {
            let t = render_texture(mood, 2.0, 5);
            assert!(
                t.lr_correlation().abs() < 0.5,
                "{} texture corr {}",
                mood.name(),
                t.lr_correlation()
            );
            assert!(t.rms() > 0.005);
        }
    }

    #[test]
    fn beds_loop_without_a_seam_or_a_periodic_dip() {
        let bed = render_bed(AmbienceMood::ServerHum, 4.0, 3);
        let n = bed.len();
        // The seam step is no bigger than an ordinary step inside the loop.
        let seam = (bed.l[n - 1] - bed.l[0]).abs();
        let max_step = bed
            .l
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            seam <= max_step * 1.01,
            "seam {seam} vs max step {max_step}"
        );
        // No envelope fade at the ends: the first and last 0.2 s are as loud as
        // the middle (the old beds faded to silence here every loop).
        let w = secs(0.2);
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let head = rms(&bed.l[..w]);
        let mid = rms(&bed.l[n / 2..n / 2 + w]);
        assert!(head > 0.5 * mid, "loop head {head} dips vs middle {mid}");
    }

    #[test]
    fn rendering_is_deterministic() {
        assert_eq!(
            render_bed(AmbienceMood::ColdVoid, 1.0, 9),
            render_bed(AmbienceMood::ColdVoid, 1.0, 9)
        );
        assert_ne!(
            render_bed(AmbienceMood::ColdVoid, 1.0, 9),
            render_bed(AmbienceMood::ColdVoid, 1.0, 10)
        );
        for k in LoopKind::ALL {
            assert_eq!(render_loop(k), render_loop(k));
        }
        for k in DetailKind::ALL {
            assert_eq!(render_detail(k, 4), render_detail(k, 4));
        }
    }

    #[test]
    fn harmony_fills_in_with_progress_and_tension_fades() {
        let mut prev = harmony_gains(0.0, 0.0);
        for step in 1..=20 {
            let p = step as f32 / 20.0;
            let g = harmony_gains(p, 0.0);
            for v in 0..TENSION_VOICE {
                assert!(g[v] + 1e-6 >= prev[v], "consonant voice {v} fell at p={p}");
            }
            assert!(
                g[TENSION_VOICE] <= prev[TENSION_VOICE] + 1e-6,
                "tension rose at p={p}"
            );
            prev = g;
        }
        let empty = harmony_gains(0.0, 0.0);
        let solved = harmony_gains(1.0, 0.0);
        // Unsolved: a bare root against the minor second.
        assert!(empty[TENSION_VOICE] > 0.3);
        assert!(empty[1..TENSION_VOICE].iter().all(|&g| g < 1e-6));
        // Solved: every consonant voice sounding, no dissonance.
        assert!(solved[..TENSION_VOICE].iter().all(|&g| g > 0.4));
        assert!(solved[TENSION_VOICE] < 1e-6);
        // A refusal spike brings the clash back even in a solved room.
        assert!(harmony_gains(1.0, 1.0)[TENSION_VOICE] > 0.8);
    }

    #[test]
    fn harmony_voices_are_the_consonant_chord_plus_one_clash() {
        let f = harmony_freqs(AmbienceMood::OrganicDecay);
        assert!((f[1] / f[0] - 1.5).abs() < 1e-5);
        assert!((f[2] / f[0] - 2.0).abs() < 1e-5);
        assert!((f[TENSION_VOICE] / f[0] - 16.0 / 15.0).abs() < 1e-5);
        let v = render_harmony_voice(f[0], 2.0, 1);
        assert!(v.rms() > 0.05);
        assert_ne!(v.l, v.r);
    }

    #[test]
    fn tension_spikes_and_decays() {
        let mut t = Tension::default();
        t.pulse(1.0);
        assert_eq!(t.level(), 1.0);
        t.decay(Tension::HALF_LIFE);
        assert!((t.level() - 0.5).abs() < 1e-4);
        t.pulse(0.3); // smaller spike does not lower it
        assert!((t.level() - 0.5).abs() < 1e-4);
        t.decay(60.0);
        assert_eq!(t.level(), 0.0);
    }

    #[test]
    fn progress_snaps_to_one_when_solved() {
        assert_eq!(harmony_progress(1, 4, false), 0.25);
        assert_eq!(harmony_progress(1, 4, true), 1.0);
        assert_eq!(harmony_progress(0, 0, false), 0.0);
    }

    #[test]
    fn detail_scheduler_is_sparse_irregular_and_deterministic() {
        let run = |seed| {
            let mut s = DetailScheduler::new(AmbienceMood::ServerHum, seed);
            let mut evs = Vec::new();
            let mut t = 0.0f32;
            while t < 120.0 {
                while let Some(e) = s.poll(t) {
                    evs.push((t, e));
                }
                t += 0.05;
            }
            evs
        };
        let a = run(3);
        assert_eq!(a, run(3));
        // ~4.5 s mean gap over two minutes → a sparse handful, not a stream.
        assert!((12..=45).contains(&a.len()), "{} events", a.len());
        // Irregular: not all gaps equal.
        let gaps: Vec<f32> = a.windows(2).map(|w| w[1].0 - w[0].0).collect();
        let (mn, mx) = gaps
            .iter()
            .fold((f32::MAX, 0.0f32), |(a, b), &g| (a.min(g), b.max(g)));
        assert!(mx > mn * 1.5, "gaps too regular: {mn}..{mx}");
        // Spread across the field, both sides.
        assert!(a.iter().any(|e| e.1.pan < -0.3) && a.iter().any(|e| e.1.pan > 0.3));
        // ServerHum favours relays / chirps; it never picks a kind weighted 0.
        assert!(a
            .iter()
            .any(|e| matches!(e.1.kind, DetailKind::RelayClick | DetailKind::DataChirp)));
    }

    #[test]
    fn loops_are_audible_and_bounded() {
        for k in LoopKind::ALL {
            let l = render_loop(k);
            assert_eq!(l.len(), secs(LOOP_SECS));
            let peak = l.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!((peak - 0.5).abs() < 1e-3, "{k:?} peak {peak}");
        }
    }

    #[test]
    fn pipe_hum_strains_with_load_and_node_glow_climbs() {
        assert!(pipe_hum_rate(1.0) > pipe_hum_rate(0.5) && pipe_hum_rate(0.5) > pipe_hum_rate(0.0));
        assert!(pipe_hum_level(1.0) > pipe_hum_level(0.0));
        let mut prev = 0.0;
        for i in 0..12 {
            let r = node_glow_rate(i);
            assert!(r > prev, "node {i}");
            prev = r;
        }
    }

    #[test]
    fn scene_bakes_the_room_into_every_layer() {
        let small = Room::from_cells(5, 5, 3, false, 0.3);
        let large = Room::from_cells(28, 24, 8, false, 0.1);
        let mk = |room| SceneSpec {
            bed_secs: 2.0,
            harmony_secs: 2.0,
            ..SceneSpec::new(AmbienceMood::ColdVoid, room, 1)
        };
        let a = SceneAssets::render(mk(small));
        assert_eq!(a.harmony.len(), HARMONY_VOICES);
        assert_eq!(a.details.len(), DetailKind::ALL.len());
        assert!(a.bed.peak() <= 0.551 && a.texture.peak() <= 0.351);
        // A detail event rings longer in the big room.
        let b = SceneAssets::render(mk(large));
        let drip = DetailKind::Drip.index();
        assert!(b.details[drip].len() > a.details[drip].len());
    }
}
