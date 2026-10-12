//! `offline` — a deterministic, device-free mixdown of the soundscape to stereo
//! PCM / WAV. **Pure compute, no device.**
//!
//! The kira path cannot be heard in a headless test, and the owner asked for
//! silence while this was built — so this module renders what the device plays,
//! with the same layer levels ([`crate::soundscape`] mix constants), the same
//! spatialization ([`crate::spatial`]), the same cue table ([`crate::cue`]) and
//! the same room ([`crate::acoustics::RoomVerb`] standing in for kira's master
//! reverb), into a buffer that can be inspected by tests or written to a WAV for
//! a human to audition.
//!
//! What differs from the device: kira's own reverb is a different algorithm from
//! [`RoomVerb`] (the one-shot / positional tail will sound slightly different),
//! and live tweens are approximated by per-block linear ramps. The ambience
//! layers are identical — they are baked by the same code on both paths.

use std::io::Write;
use std::path::Path;

use crate::acoustics::{Room, RoomVerb};
use crate::cue::Cue;
use crate::dsp::{pan_gains, resample, secs, OnePole, Stereo, VariableDelay};
use crate::soundscape::{
    harmony_gains, DetailScheduler, SceneAssets, DETAIL_LEVEL, HARMONY_LEVEL, TEXTURE_LEVEL,
};
use crate::spatial::{select_voices, Listener, LoopKind, PositionalSource};
use crate::synth::{SynthBank, SAMPLE_RATE};

const BLOCK: usize = 256;
/// Capacity of each positional voice's ITD ring buffer (samples). Comfortably
/// covers [`crate::spatial::MAX_ITD_MS`] (< 1 ms ⇒ well under 48 samples at
/// [`SAMPLE_RATE`]) with margin.
const ITD_LINE_CAPACITY: usize = 48;

/// A stereo mix bus with a dry path and a room ("fog") send.
pub struct OfflineMix {
    dry: Stereo,
    room_send: Stereo,
}

impl OfflineMix {
    /// A silent mix `secs_len` long.
    pub fn new(secs_len: f32) -> OfflineMix {
        let n = secs(secs_len);
        OfflineMix {
            dry: Stereo::silent(n),
            room_send: Stereo::silent(n),
        }
    }

    /// Length in frames.
    pub fn len(&self) -> usize {
        self.dry.len()
    }

    /// `true` for an empty mix.
    pub fn is_empty(&self) -> bool {
        self.dry.is_empty()
    }

    /// Tile a looping stereo layer across the whole mix, its gain given per
    /// block by `gain_at(seconds)` (linearly ramped between blocks, like a tween).
    pub fn add_loop(&mut self, src: &Stereo, gain_at: impl Fn(f32) -> f32) {
        if src.is_empty() {
            return;
        }
        let n = self.len();
        let mut g0 = gain_at(0.0);
        let mut i = 0;
        while i < n {
            let end = (i + BLOCK).min(n);
            let g1 = gain_at(end as f32 / SAMPLE_RATE as f32);
            for j in i..end {
                let t = (j - i) as f32 / (end - i) as f32;
                let g = g0 + (g1 - g0) * t;
                let k = j % src.len();
                self.dry.l[j] += src.l[k] * g;
                self.dry.r[j] += src.r[k] * g;
            }
            g0 = g1;
            i = end;
        }
    }

    /// Place a stereo one-shot at `at_s`, re-pitched by `rate`, panned and
    /// scaled. `room` sends it through the room bus (the device's master fog).
    pub fn add_stereo_oneshot(
        &mut self,
        at_s: f32,
        src: &Stereo,
        rate: f64,
        gain: f32,
        pan: f32,
        room: bool,
    ) {
        let l = resample(&src.l, rate);
        let r = resample(&src.r, rate);
        let (gl, gr) = pan_gains(pan);
        // Normalise the pan law so centre is unity (kira's convention).
        let (gl, gr) = (gl * std::f32::consts::SQRT_2, gr * std::f32::consts::SQRT_2);
        let start = secs(at_s);
        let bus = if room {
            &mut self.room_send
        } else {
            &mut self.dry
        };
        for (k, (a, b)) in l.iter().zip(r.iter()).enumerate() {
            let j = start + k;
            if j >= bus.len() {
                break;
            }
            bus.l[j] += a * gain * gl.min(1.0);
            bus.r[j] += b * gain * gr.min(1.0);
        }
    }

    /// Place a mono one-shot (a synth bank sound).
    pub fn add_oneshot(&mut self, at_s: f32, src: &[f32], rate: f64, gain: f32, pan: f32) {
        self.add_stereo_oneshot(at_s, &Stereo::from_mono(src), rate, gain, pan, true);
    }

    /// Voice a [`Cue`] at `at_s` through the cue table, optionally from a world
    /// position heard by `listener`.
    pub fn add_cue(
        &mut self,
        bank: &SynthBank,
        at_s: f32,
        cue: Cue,
        from: Option<([f32; 3], Listener)>,
    ) {
        let placed = cue.placed_voices(from.as_ref().map(|(p, l)| (*p, l)));
        for v in placed {
            self.add_oneshot(
                at_s,
                &bank.get(v.sound).frames,
                v.rate,
                v.gain * CUE_LEVEL,
                v.pan,
            );
        }
    }

    /// Play positional loops for a moving listener: `path(t)` gives the
    /// listener and the source list at time `t`; at most `max_voices` sound.
    ///
    /// Applies [`crate::spatial::VoiceMix::cutoff_hz`] (air absorption +
    /// occlusion + front/back muffling, per-channel one-pole low-pass,
    /// live-retuned as it changes) and
    /// [`crate::spatial::VoiceMix::itd_ms`] (a small variable delay so the far ear genuinely
    /// lags the near one — this is the one cue only the offline path renders
    /// exactly; see `crate::spatial`'s module doc for why the device path
    /// does not attempt it).
    pub fn add_positional(
        &mut self,
        loops: &[Vec<f32>; 4],
        max_voices: usize,
        path: impl Fn(f32) -> (Listener, Vec<PositionalSource>),
    ) {
        let n = self.len();
        // Per-voice playback cursors (fractional, for the rate).
        let mut cursors: std::collections::HashMap<crate::spatial::SourceKey, f64> =
            std::collections::HashMap::new();
        let mut prev: std::collections::HashMap<crate::spatial::SourceKey, (f32, f32)> =
            std::collections::HashMap::new();
        // Per-voice cutoff filter state (L, R) — persists so retuning never clicks.
        let mut filt: std::collections::HashMap<crate::spatial::SourceKey, (OnePole, OnePole)> =
            std::collections::HashMap::new();
        // Per-voice ITD line — a shared mono history each ear reads at its own lag.
        let mut itd_line: std::collections::HashMap<crate::spatial::SourceKey, VariableDelay> =
            std::collections::HashMap::new();
        let mut i = 0;
        while i < n {
            let end = (i + BLOCK).min(n);
            let (l, sources) = path(i as f32 / SAMPLE_RATE as f32);
            let voices = select_voices(&l, &sources, max_voices);
            let mut next_prev = std::collections::HashMap::new();
            for vm in &voices {
                let src = &loops[vm.key.kind.index()];
                let (gl1, gr1) = pan_gains(vm.pan);
                let target = (
                    gl1 * vm.gain * POSITIONAL_LEVEL,
                    gr1 * vm.gain * POSITIONAL_LEVEL,
                );
                let start = prev.get(&vm.key).copied().unwrap_or((0.0, 0.0));
                let cur = cursors.entry(vm.key).or_insert(0.0);
                let (fl, fr) = filt.entry(vm.key).or_insert_with(|| {
                    (
                        OnePole::lowpass(vm.cutoff_hz),
                        OnePole::lowpass(vm.cutoff_hz),
                    )
                });
                fl.retune(vm.cutoff_hz);
                fr.retune(vm.cutoff_hz);
                let line = itd_line
                    .entry(vm.key)
                    .or_insert_with(|| VariableDelay::new(ITD_LINE_CAPACITY));
                let itd_samples =
                    ((vm.itd_ms.abs() / 1000.0) * SAMPLE_RATE as f32).round() as usize;
                // itd_ms > 0 ⇒ left lags (source on the right); < 0 ⇒ right lags.
                let (delay_l, delay_r) = if vm.itd_ms >= 0.0 {
                    (itd_samples, 0)
                } else {
                    (0, itd_samples)
                };
                for j in i..end {
                    let t = (j - i) as f32 / (end - i) as f32;
                    let gl = start.0 + (target.0 - start.0) * t;
                    let gr = start.1 + (target.1 - start.1) * t;
                    let k = (*cur as usize) % src.len();
                    line.push(src[k]);
                    let sl = fl.lp(line.read(delay_l));
                    let sr = fr.lp(line.read(delay_r));
                    self.room_send.l[j] += sl * gl;
                    self.room_send.r[j] += sr * gr;
                    *cur += vm.rate;
                }
                next_prev.insert(vm.key, target);
            }
            prev = next_prev;
            i = end;
        }
    }

    /// Sum the dry path with the room bus processed through `room`, then soft
    /// limit into `[-1, 1]`.
    pub fn finish(self, room: &Room) -> Stereo {
        let n = self.dry.len();
        let mut verb = RoomVerb::new(room);
        let mut out = self.dry;
        for i in 0..n {
            let (l, r) = verb.tick(self.room_send.l[i], self.room_send.r[i]);
            out.l[i] += l;
            out.r[i] += r;
        }
        for v in out.l.iter_mut().chain(out.r.iter_mut()) {
            // tanh-style soft clip: transparent below ~0.5, never exceeds 1.
            *v = v.tanh();
        }
        out
    }
}

/// Level of a cue one-shot relative to the ambience (mirrors the device).
pub const CUE_LEVEL: f32 = 0.8;
/// Level of the positional loops (mirrors the device).
pub use crate::soundscape::POSITIONAL_LEVEL;

/// Add the full ambience of `scene` to `mix`: bed + texture + harmony (driven by
/// `progress_at(t)` and `tension_at(t)`) + seeded detail events.
pub fn add_ambience(
    mix: &mut OfflineMix,
    scene: &SceneAssets,
    intensity_gain: f32,
    progress_at: impl Fn(f32) -> f32,
    tension_at: impl Fn(f32) -> f32,
) {
    mix.add_loop(&scene.bed, |_| intensity_gain);
    mix.add_loop(&scene.texture, |_| intensity_gain * TEXTURE_LEVEL);
    for (v, voice) in scene.harmony.iter().enumerate() {
        mix.add_loop(voice, |t| {
            harmony_gains(progress_at(t), tension_at(t))[v] * HARMONY_LEVEL
        });
    }
    let mut sched = DetailScheduler::new(scene.spec.mood, scene.spec.seed);
    let dur = mix.len() as f32 / SAMPLE_RATE as f32;
    let mut t = 0.0f32;
    while t < dur {
        while let Some(e) = sched.poll(t) {
            let d = &scene.details[e.kind.index()];
            mix.add_stereo_oneshot(t, d, e.rate, e.gain * DETAIL_LEVEL, e.pan, false);
        }
        t += 0.05;
    }
}

/// Write a 16-bit PCM stereo WAV at [`SAMPLE_RATE`].
pub fn write_wav(path: &Path, s: &Stereo) -> std::io::Result<()> {
    let n = s.len() as u32;
    let data_len = n * 4;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&2u16.to_le_bytes())?; // stereo
    f.write_all(&SAMPLE_RATE.to_le_bytes())?;
    f.write_all(&(SAMPLE_RATE * 4).to_le_bytes())?;
    f.write_all(&4u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for i in 0..s.len() {
        for v in [s.l[i], s.r[i]] {
            let q = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            f.write_all(&q.to_le_bytes())?;
        }
    }
    f.flush()
}

/// The four positional loops, rendered once.
pub fn positional_loops() -> [Vec<f32>; 4] {
    LoopKind::ALL.map(crate::soundscape::render_loop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ambience::AmbienceMood;
    use crate::soundscape::SceneSpec;
    use crate::spatial::SourceKey;

    fn short_scene() -> SceneAssets {
        SceneAssets::render(SceneSpec {
            bed_secs: 2.0,
            harmony_secs: 2.0,
            ..SceneSpec::new(AmbienceMood::ServerHum, Room::default(), 3)
        })
    }

    #[test]
    fn ambience_mix_is_stereo_bounded_and_deterministic() {
        let scene = short_scene();
        let render = || {
            let mut m = OfflineMix::new(3.0);
            add_ambience(&mut m, &scene, 0.7, |_| 0.4, |_| 0.0);
            m.finish(&Room::default())
        };
        let a = render();
        assert_eq!(a, render());
        assert!(a.peak() <= 1.0);
        assert!(a.rms() > 0.01);
        assert!(
            a.lr_correlation() < 0.95,
            "mixdown is mono-ish: {}",
            a.lr_correlation()
        );
    }

    #[test]
    fn progress_is_audible_in_the_ambience() {
        // The same room at 0 % vs 100 % progress must differ, and the solved
        // mix carries more harmonic energy.
        let scene = short_scene();
        let at = |p: f32| {
            let mut m = OfflineMix::new(2.0);
            add_ambience(&mut m, &scene, 0.0, move |_| p, |_| 0.0);
            m.finish(&Room::default())
        };
        let (lo, hi) = (at(0.0), at(1.0));
        assert_ne!(lo, hi);
        assert!(
            hi.rms() > lo.rms(),
            "solved chord {} not fuller than bare root {}",
            hi.rms(),
            lo.rms()
        );
    }

    #[test]
    fn a_lamp_on_the_right_is_louder_in_the_right_ear() {
        let loops = positional_loops();
        let mut m = OfflineMix::new(1.0);
        m.add_positional(&loops, 4, |_| {
            (
                Listener {
                    pos: [0.0, 0.0, 1.5],
                    yaw: 0.0,
                },
                vec![PositionalSource {
                    key: SourceKey {
                        kind: LoopKind::LampHum,
                        id: 0,
                    },
                    pos: [2.0, 0.5, 1.5],
                    level: 1.0,
                    rate: 1.0,
                    occlusion: 0.0,
                }],
            )
        });
        let s = m.finish(&Room {
            absorption: 0.9,
            ..Room::default()
        });
        let e = |c: &[f32]| c.iter().map(|v| v * v).sum::<f32>();
        assert!(
            e(&s.r) > 2.0 * e(&s.l),
            "right {} vs left {}",
            e(&s.r),
            e(&s.l)
        );
    }

    #[test]
    fn cues_land_where_they_are_scheduled() {
        let bank = SynthBank::build();
        let mut m = OfflineMix::new(1.0);
        m.add_cue(&bank, 0.5, Cue::SwitchOn { used: 1, budget: 3 }, None);
        let s = m.finish(&Room {
            absorption: 0.95,
            open_sky: true,
            ..Room::default()
        });
        let first = s.l.iter().position(|v| v.abs() > 1e-3).unwrap();
        assert!(
            (first as i64 - secs(0.5) as i64).abs() < 200,
            "cue started at {first}"
        );
    }

    #[test]
    fn wav_writer_emits_a_valid_header() {
        let dir = std::env::temp_dir().join(format!("re_audio_wav_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wav");
        write_wav(&p, &Stereo::silent(100)).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[..4], b"RIFF");
        assert_eq!(&b[8..16], b"WAVEfmt ");
        assert_eq!(b.len(), 44 + 400);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
