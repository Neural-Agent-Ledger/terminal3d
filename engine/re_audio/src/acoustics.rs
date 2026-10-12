//! `acoustics` — the level's ROOM as an acoustic object. **Pure logic, no device.**
//!
//! A level is a box of voxels with known dimensions, an optional open sky and
//! surfaces made of named materials. That is exactly the information a simple
//! geometric room model needs, so the ambience stops being a dry mono bed played
//! "nowhere" and is instead heard *in this room*:
//!
//! * [`Room`] — size in metres, open sky or not, mean surface absorption. The
//!   game builds one from `re_level` data (`[room] size`, `[sky]`, the floor /
//!   wall `[material]` presets) and sends it with
//!   [`SetRoom`](crate::plugin::SetRoom); `re_audio` never depends on `re_level`.
//! * [`Room::rt60`] — Sabine reverberation time from volume and absorbing area
//!   (an open sky is a perfectly absorbing ceiling).
//! * [`Room::early_reflections`] — first- and second-order image-source taps
//!   (delay, gain, pan) for a listener standing in the room: the left wall comes
//!   back from the left, the far wall later than the near one. These are what the
//!   ear uses to judge room size, and they are what makes the bed wide.
//! * [`RoomVerb`] — a deterministic stereo early-reflection + Freeverb-style
//!   tail processor, used to **bake** the room into the looping ambience layers
//!   (so they are spatial on the device and in the offline WAV renders alike).
//! * [`Room::kira_reverb`] — the matching parameters for kira's master-fog
//!   reverb, which carries the room for one-shots and positional sources.
//!
//! Determinism: fixed delay tables and no RNG — the same room always bakes the
//! same samples.

use crate::dsp::{pan_gains, Allpass, Comb, Delay, OnePole, Stereo};
use crate::synth::SAMPLE_RATE;

/// World cell edge in metres. The raycaster's cell is roughly a person's
/// shoulder-width-and-a-half; 1.2 m keeps a 16-cell hall at ~19 m.
pub const METERS_PER_CELL: f32 = 1.2;
/// Speed of sound in air (m/s).
pub const SPEED_OF_SOUND: f32 = 343.0;
/// Listener ear height (m) used for the floor / ceiling image sources.
const EAR_HEIGHT_M: f32 = 1.6;

/// The acoustic room a level describes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room {
    /// Interior extent `[width (x), depth (y), height (z)]` in metres.
    pub size_m: [f32; 3],
    /// `true` when the level has a sky: there is no ceiling to reflect from.
    pub open_sky: bool,
    /// Mean surface absorption coefficient in `[0.02, 0.95]` (hard metal/stone
    /// ≈ 0.05–0.15, concrete ≈ 0.2, snow/organic/fabric ≈ 0.5+).
    pub absorption: f32,
}

impl Default for Room {
    /// A mid-sized concrete room (12 × 12 × 4.8 m): the neutral fallback.
    fn default() -> Self {
        Room {
            size_m: [12.0, 12.0, 4.8],
            open_sky: false,
            absorption: 0.22,
        }
    }
}

/// One early-reflection tap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tap {
    /// Arrival delay relative to the direct sound (seconds).
    pub delay_s: f32,
    /// Linear gain (already includes absorption and spreading loss).
    pub gain: f32,
    /// Where the reflection arrives from, `[-1, 1]` (−1 = left).
    pub pan: f32,
}

/// Parameters for kira's reverb effect (the device-side tail for one-shots).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KiraReverb {
    /// Tail length control in `[0, 1)`.
    pub feedback: f64,
    /// High-frequency damping in `[0, 1]`.
    pub damping: f64,
    /// Wet mix in `[0, 1]`.
    pub mix: f32,
}

impl Room {
    /// Build a room from voxel dimensions (cells) — what a `.lvl` authors.
    pub fn from_cells(sx: usize, sy: usize, sz: usize, open_sky: bool, absorption: f32) -> Room {
        let m = |c: usize| (c.max(1) as f32) * METERS_PER_CELL;
        Room {
            size_m: [m(sx), m(sy), m(sz)],
            open_sky,
            absorption: absorption.clamp(0.02, 0.95),
        }
    }

    /// Interior volume (m³).
    pub fn volume_m3(&self) -> f32 {
        self.size_m[0] * self.size_m[1] * self.size_m[2]
    }

    /// Sabine RT60 (seconds): `0.161·V / A`, where the open-sky ceiling absorbs
    /// everything that reaches it. Clamped to `[0.15, 6.0]`.
    pub fn rt60(&self) -> f32 {
        let [w, d, h] = self.size_m;
        let floor = w * d;
        let walls = 2.0 * (w * h + d * h);
        let a = self.absorption;
        let ceiling_a = if self.open_sky { 1.0 } else { a };
        let absorbing = floor * a + walls * a + floor * ceiling_a;
        (0.161 * self.volume_m3() / absorbing.max(1e-3)).clamp(0.15, 6.0)
    }

    /// Image-source early reflections for a listener standing slightly off the
    /// room's centre (40 % across, 45 % deep — never perfectly symmetric, which
    /// would collapse the L/R taps onto the same delay). First-order: six
    /// surfaces (five with an open sky); second-order: the four vertical
    /// corner paths. Taps later than 150 ms are dropped (that is tail, not early
    /// reflection).
    pub fn early_reflections(&self) -> Vec<Tap> {
        let [w, d, h] = self.size_m;
        let (lx, ly) = (0.40 * w, 0.45 * d);
        let ez = EAR_HEIGHT_M.min(h * 0.6);
        let refl = 1.0 - self.absorption;
        let mut taps = Vec::with_capacity(10);
        // (extra path length in metres, number of bounces, pan)
        let mut push = |extra: f32, bounces: i32, pan: f32| {
            let delay = extra / SPEED_OF_SOUND;
            if delay > 0.15 || extra <= 0.0 {
                return;
            }
            // Spreading loss relative to a 1 m direct path, plus absorption per
            // bounce.
            let gain = refl.powi(bounces) / (1.0 + extra / 2.0);
            taps.push(Tap {
                delay_s: delay,
                gain,
                pan,
            });
        };
        push(2.0 * lx, 1, -0.85); // left wall
        push(2.0 * (w - lx), 1, 0.85); // right wall
        push(2.0 * ly, 1, -0.20); // near wall (behind): slightly left
        push(2.0 * (d - ly), 1, 0.20); // far wall (ahead): slightly right
        push(2.0 * ez, 1, 0.0); // floor
        if !self.open_sky {
            push(2.0 * (h - ez), 1, 0.05); // ceiling
        }
        // Second-order vertical corners (a wall then the adjacent wall).
        let corner = |a: f32, b: f32| (a * a + b * b).sqrt() * 2.0;
        push(corner(lx, ly), 2, -0.6);
        push(corner(w - lx, ly), 2, 0.6);
        push(corner(lx, d - ly), 2, -0.45);
        push(corner(w - lx, d - ly), 2, 0.45);
        taps.sort_by(|a, b| a.delay_s.total_cmp(&b.delay_s));
        taps
    }

    /// The wet level of the diffuse tail: bigger, harder rooms are wetter; an
    /// open sky lets most of the energy escape.
    pub fn tail_mix(&self) -> f32 {
        let rt = self.rt60();
        let mut mix = 0.10 + 0.32 * (1.0 - (-rt / 1.4).exp());
        if self.open_sky {
            mix *= 0.55;
        }
        mix.clamp(0.05, 0.45)
    }

    /// kira reverb parameters carrying this room for the one-shot / positional
    /// path (the ambience layers have the room baked in by [`RoomVerb`]).
    pub fn kira_reverb(&self) -> KiraReverb {
        let rt = self.rt60() as f64;
        KiraReverb {
            feedback: (0.62 + 0.35 * (1.0 - (-rt / 1.6).exp())).clamp(0.55, 0.97),
            damping: (0.15 + 0.8 * self.absorption as f64).clamp(0.05, 0.95),
            mix: (self.tail_mix() * 0.8).clamp(0.05, 0.40),
        }
    }
}

/// Freeverb comb / all-pass tunings (samples at 44.1 kHz) and the stereo
/// spread added to the right channel's lines — decorrelated L/R tails.
const COMB_TUNING: [usize; 4] = [1116, 1277, 1422, 1557];
const ALLPASS_TUNING: [usize; 2] = [556, 441];
const STEREO_SPREAD: usize = 23;

/// A deterministic stereo room processor: early-reflection taps + a
/// Freeverb-style diffuse tail whose length follows the room's RT60.
///
/// The input is mono or stereo *dry* signal; the output is the dry signal plus
/// the room. Feed it the same buffer twice ([`RoomVerb::bake_loop`]) to get a
/// seamlessly looping wet buffer whose tail wraps around the loop point.
pub struct RoomVerb {
    taps: Vec<(Delay, f32, f32, f32)>, // (line, gain, gl, gr)
    combs_l: Vec<Comb>,
    combs_r: Vec<Comb>,
    ap_l: Vec<Allpass>,
    ap_r: Vec<Allpass>,
    predelay: Delay,
    tail_mix: f32,
    er_mix: f32,
    tone: (OnePole, OnePole),
}

impl RoomVerb {
    /// Build the processor for `room`.
    pub fn new(room: &Room) -> RoomVerb {
        let sr = SAMPLE_RATE as f32;
        let taps = room
            .early_reflections()
            .into_iter()
            .map(|t| {
                let (gl, gr) = pan_gains(t.pan);
                (
                    Delay::new((t.delay_s * sr).round() as usize),
                    t.gain,
                    gl,
                    gr,
                )
            })
            .collect();
        // Bigger rooms → longer comb lines (the modal density grows with size).
        let scale = (room.volume_m3().cbrt() / 10.0).clamp(0.55, 1.6);
        let rt = room.rt60();
        let damp = (0.10 + 0.75 * room.absorption).clamp(0.05, 0.9);
        let mk = |spread: usize| -> Vec<Comb> {
            COMB_TUNING
                .iter()
                .map(|&n| {
                    let len = ((n as f32 * scale) as usize + spread).max(64);
                    // Per-comb feedback for a −60 dB decay in `rt` seconds.
                    let g = 10f32.powf(-3.0 * len as f32 / (rt * sr));
                    Comb::new(len, g, damp)
                })
                .collect()
        };
        let ap = |spread: usize| -> Vec<Allpass> {
            ALLPASS_TUNING
                .iter()
                .map(|&n| Allpass::new(n + spread, 0.5))
                .collect()
        };
        // Pre-delay: the gap before the diffuse field arrives ≈ the first
        // reflection of the far wall.
        let pre = (room.size_m[0].max(room.size_m[1]) / SPEED_OF_SOUND * 0.5 * sr) as usize;
        // Absorbent rooms return a darker field.
        let tone_hz = 9000.0 - 6500.0 * room.absorption;
        RoomVerb {
            taps,
            combs_l: mk(0),
            combs_r: mk(STEREO_SPREAD),
            ap_l: ap(0),
            ap_r: ap(STEREO_SPREAD),
            predelay: Delay::new(pre.max(1)),
            tail_mix: room.tail_mix(),
            er_mix: 0.55,
            tone: (OnePole::lowpass(tone_hz), OnePole::lowpass(tone_hz)),
        }
    }

    /// Process one stereo frame; returns `(l, r)` = dry + early + tail.
    #[inline]
    pub fn tick(&mut self, xl: f32, xr: f32) -> (f32, f32) {
        let mono = 0.5 * (xl + xr);
        let (mut el, mut er) = (0.0f32, 0.0f32);
        for (line, g, gl, gr) in &mut self.taps {
            // A wall reflects what reaches it: the left wall's echo carries
            // mostly the left-side content (weights gl², gr² sum to 1, so a
            // centred source feeds every tap exactly its mono level). Feeding
            // the taps a plain mono sum instead would stamp the room's
            // geometry bias onto every source and drag a source on the right
            // towards the nearer left wall.
            let v = line.tick(*gl * *gl * xl + *gr * *gr * xr) * *g;
            el += v * *gl;
            er += v * *gr;
        }
        let into_tail = self.predelay.tick(mono + 0.5 * (el + er)) * 0.25;
        let mut tl = 0.0f32;
        let mut tr = 0.0f32;
        for c in &mut self.combs_l {
            tl += c.tick(into_tail);
        }
        for c in &mut self.combs_r {
            tr += c.tick(into_tail);
        }
        for a in &mut self.ap_l {
            tl = a.tick(tl);
        }
        for a in &mut self.ap_r {
            tr = a.tick(tr);
        }
        let tl = self.tone.0.lp(tl);
        let tr = self.tone.1.lp(tr);
        (
            xl + self.er_mix * el + self.tail_mix * tl,
            xr + self.er_mix * er + self.tail_mix * tr,
        )
    }

    /// Process a whole (non-looping) stereo buffer, appending `tail_secs` of
    /// ring-out so a one-shot's reverb is not cut off.
    pub fn process(&mut self, dry: &Stereo, tail_secs: f32) -> Stereo {
        let extra = crate::dsp::secs(tail_secs);
        let n = dry.len() + extra;
        let mut out = Stereo::silent(n);
        for i in 0..n {
            let (xl, xr) = if i < dry.len() {
                (dry.l[i], dry.r[i])
            } else {
                (0.0, 0.0)
            };
            let (l, r) = self.tick(xl, xr);
            out.l[i] = l;
            out.r[i] = r;
        }
        out
    }

    /// Bake the room into a seamlessly-looping stereo buffer: run the loop
    /// through twice and keep the second pass, whose reverb history is the end of
    /// the loop itself — so the tail wraps across the loop point with no seam.
    pub fn bake_loop(mut self, dry: &Stereo) -> Stereo {
        let n = dry.len();
        let mut out = Stereo::silent(n);
        for pass in 0..2 {
            for i in 0..n {
                let (l, r) = self.tick(dry.l[i], dry.r[i]);
                if pass == 1 {
                    out.l[i] = l;
                    out.r[i] = r;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Room {
        Room::from_cells(5, 5, 3, false, 0.25)
    }
    fn large() -> Room {
        Room::from_cells(28, 24, 8, false, 0.12)
    }

    #[test]
    fn bigger_harder_rooms_ring_longer() {
        assert!(
            large().rt60() > small().rt60() * 2.0,
            "{} vs {}",
            large().rt60(),
            small().rt60()
        );
        // Absorption shortens the tail.
        let soft = Room {
            absorption: 0.6,
            ..large()
        };
        assert!(soft.rt60() < large().rt60());
        // An open sky shortens it too.
        let sky = Room {
            open_sky: true,
            ..large()
        };
        assert!(sky.rt60() < large().rt60());
    }

    #[test]
    fn early_reflections_follow_geometry() {
        let s = small().early_reflections();
        let l = large().early_reflections();
        assert!(!s.is_empty() && !l.is_empty());
        // The walls of a big room answer later (the floor tap is the same
        // height in both, so compare the lateral reflections and the mean).
        let first_wall = |t: &[Tap]| t.iter().find(|x| x.pan.abs() > 0.5).unwrap().delay_s;
        assert!(first_wall(&l) > first_wall(&s));
        let mean = |t: &[Tap]| t.iter().map(|x| x.delay_s).sum::<f32>() / t.len() as f32;
        assert!(mean(&l) > mean(&s));
        // Both sides are represented (a wide image), and the left wall comes
        // from the left.
        assert!(s.iter().any(|t| t.pan < -0.5) && s.iter().any(|t| t.pan > 0.5));
        // Sorted by arrival, all within the early window.
        for w in s.windows(2) {
            assert!(w[0].delay_s <= w[1].delay_s);
        }
        assert!(l
            .iter()
            .all(|t| t.delay_s <= 0.15 && t.gain > 0.0 && t.gain < 1.0));
        // An open sky removes the ceiling tap.
        let sky = Room {
            open_sky: true,
            ..small()
        };
        assert_eq!(
            sky.early_reflections().len() + 1,
            small().early_reflections().len()
        );
    }

    fn impulse_energy_after(room: &Room, after_s: f32) -> f32 {
        let mut dry = Stereo::silent(10);
        dry.l[0] = 1.0;
        dry.r[0] = 1.0;
        let wet = RoomVerb::new(room).process(&dry, 2.5);
        let from = crate::dsp::secs(after_s);
        wet.l[from..]
            .iter()
            .chain(wet.r[from..].iter())
            .map(|v| v * v)
            .sum()
    }

    #[test]
    fn a_large_room_tail_outlasts_a_small_one() {
        let big = impulse_energy_after(&large(), 0.6);
        let little = impulse_energy_after(&small(), 0.6);
        assert!(
            big > little * 3.0,
            "late energy big {big} vs small {little}"
        );
    }

    #[test]
    fn the_room_decorrelates_a_mono_source() {
        // A mono noise burst in → a stereo field out (L != R).
        let mut lcg = crate::synth::Lcg(99);
        let m: Vec<f32> = (0..22_050)
            .map(|_| crate::synth::noise(&mut lcg) * 0.3)
            .collect();
        let wet = RoomVerb::new(&large()).process(&Stereo::from_mono(&m), 0.5);
        let c = wet.lr_correlation();
        assert!(c < 0.9, "room output is still mono-ish (corr {c})");
    }

    #[test]
    fn baking_is_deterministic() {
        let dry = Stereo::from_mono(
            &(0..8000)
                .map(|i| (i as f32 * 0.05).sin() * 0.2)
                .collect::<Vec<_>>(),
        );
        let a = RoomVerb::new(&small()).bake_loop(&dry);
        let b = RoomVerb::new(&small()).bake_loop(&dry);
        assert_eq!(a, b);
    }

    #[test]
    fn kira_reverb_params_are_in_range_and_follow_size() {
        for r in [
            small(),
            large(),
            Room::default(),
            Room {
                open_sky: true,
                ..large()
            },
        ] {
            let k = r.kira_reverb();
            assert!((0.0..1.0).contains(&k.feedback));
            assert!((0.0..=1.0).contains(&k.damping));
            assert!((0.0..=1.0).contains(&k.mix));
        }
        assert!(large().kira_reverb().feedback > small().kira_reverb().feedback);
    }
}
