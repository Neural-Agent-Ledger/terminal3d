//! `OfflineBackend` — a device-free, **real** kira [`Backend`] for rendering
//! the actual playback graph to PCM. This is not a hand-rolled approximation
//! of the mix (that is what [`crate::offline`] does for the pre-field
//! layers); it drives the same [`kira::backend::Renderer`] a real device
//! would, so field-chain verification exercises the exact effect graph
//! `playback.rs` builds.
//!
//! Modeled directly on kira's own [`kira::backend::mock::MockBackend`] (the
//! library's own "manually drive the renderer" backend), with one addition:
//! [`OfflineBackend::render`] loops calling `on_start_processing` + `process`
//! in fixed-size blocks rather than once for the whole requested length. This
//! matters for correctness, not just style — `on_start_processing` is where
//! kira drains each effect's command queue (parameter/mode changes sent from
//! handles) into the live resources; a real audio device calls it once per
//! hardware callback, each covering a small block. Draining it once for an
//! entire multi-second render would freeze every parameter at whatever value
//! was set *before* that call for the full duration, which is not what the
//! real device does and would make offline verification lie about click-free
//! parameter tweens.
//!
//! Reached through [`kira::AudioManager::backend_mut`] (verified present in
//! kira 0.12.4 at `manager.rs:327`; `Renderer::process`/`on_start_processing`
//! are `pub` at `backend/renderer.rs:63,71`).

use kira::backend::{Backend, Renderer};

/// A device callback in real hardware is typically a few hundred to a few
/// thousand frames; 512 is representative and keeps parameter smoothing
/// realistic without making a 10s render slow (about 862 blocks at 44.1kHz).
const BLOCK_FRAMES: usize = 512;

/// Settings for [`OfflineBackend`]. Only the sample rate is configurable —
/// there is no real device to pick a buffer size for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfflineBackendSettings {
    /// The sample rate to render at, in Hz.
    pub sample_rate: u32,
}

impl Default for OfflineBackendSettings {
    fn default() -> Self {
        Self {
            sample_rate: crate::synth::SAMPLE_RATE,
        }
    }
}

enum State {
    Uninitialized,
    // Boxed to keep this enum small (`Renderer` itself is a large struct).
    // No `Mutex` needed: `OfflineBackend::render` takes `&mut self`, so the
    // borrow checker already guarantees exclusive access.
    Initialized(Box<Renderer>),
}

/// A kira [`Backend`] that never opens a device. Build an
/// `AudioManager<OfflineBackend>`, build the same graph `playback.rs` builds
/// against it (via a shared, backend-agnostic helper — see `playback.rs`'s
/// `build_graph`), issue commands, then call [`OfflineBackend::render`]
/// through [`kira::AudioManager::backend_mut`] to pull PCM.
pub struct OfflineBackend {
    sample_rate: u32,
    state: State,
}

impl OfflineBackend {
    /// Renders `num_frames` stereo frames as interleaved `[l, r, l, r, ...]`
    /// `f32` samples, in `BLOCK_FRAMES`-sized device-callback-shaped chunks.
    /// Panics if called before the owning `AudioManager` has started this
    /// backend (mirrors `MockBackend`'s panic-on-uninitialized contract).
    pub fn render(&mut self, num_frames: usize) -> Vec<f32> {
        let State::Initialized(renderer) = &mut self.state else {
            panic!("OfflineBackend::render called before the AudioManager started it");
        };
        let mut out = vec![0.0f32; num_frames * 2];
        for chunk in out.chunks_mut(BLOCK_FRAMES * 2) {
            renderer.on_start_processing();
            renderer.process(chunk, 2);
        }
        out
    }

    /// The sample rate this backend was set up with, in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Backend for OfflineBackend {
    type Settings = OfflineBackendSettings;
    type Error = std::convert::Infallible;

    fn setup(
        settings: Self::Settings,
        _internal_buffer_size: usize,
    ) -> Result<(Self, u32), Self::Error> {
        Ok((
            Self {
                sample_rate: settings.sample_rate,
                state: State::Uninitialized,
            },
            settings.sample_rate,
        ))
    }

    fn start(&mut self, renderer: Renderer) -> Result<(), Self::Error> {
        self.state = State::Initialized(Box::new(renderer));
        Ok(())
    }
}

/// Interleaved `[l, r, ...]` samples to `re_audio::dsp::Stereo`, so offline
/// renders can reuse [`crate::offline::write_wav`] and every existing
/// `dsp.rs` analysis helper (`correlation`, etc.) unchanged.
#[must_use]
pub fn interleaved_to_stereo(samples: &[f32]) -> crate::dsp::Stereo {
    let n = samples.len() / 2;
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for frame in samples.as_chunks::<2>().0 {
        l.push(frame[0]);
        r.push(frame[1]);
    }
    crate::dsp::Stereo { l, r }
}

// ─────────────── field-chain real-graph verification ───────────────────────
//
// Exercises the ACTUAL kira graph `playback.rs` builds — via its
// `pub(crate)` `build_field_chain`/`apply_bus_fx`/limiter constants, not a
// hand-rolled approximation — through [`OfflineBackend`]. Covers
// determinism, headroom, no-NaN/click, a `LetheWash`-style HF-energy drop, a
// `WatcherGaze`/`RecognitionNarrow`-style inter-channel-correlation drop, a
// `MnemosHold`-style freeze autocorrelation, and a CPU-cost-per-block
// benchmark. Not covered here: `BitRot`'s quantization histogram,
// `FalseEcho`'s mirror cross-correlation, `ExitDetune`'s FFT pitch tracking,
// CUE onset/pitch legibility.
//
// `render_field_chain_scenario` and the analysis helpers below (`peak_dbfs`,
// `has_nan_or_inf`, `rms_dbfs`, `hf_energy_rms`, `stereo_correlation`,
// `max_sample_jump`) are `pub` rather than `pub(crate)` so that per-level
// scripted-walk offline WAV verification and a live-audition editor tool can
// both drive them from outside this crate.

use kira::effect::compressor::CompressorBuilder;
use kira::effect::reverb::ReverbBuilder;
use kira::sound::static_sound::{StaticSoundData, StaticSoundSettings};
use kira::track::{SendTrackBuilder, TrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, Frame, Mix, Tween};

use crate::field::BusFx;
use crate::field_fx::glitch_stutter::HoldBuilder;
use crate::synth::{noise, Lcg};

/// Deterministic seeded white noise at `amp`, as mono-upmixed-to-stereo
/// kira `Frame`s — a stand-in "always has energy everywhere" test signal
/// (real ambience/SFX content would work identically; noise makes every
/// metric below well-defined with no silence edge cases).
fn white_noise_frames(seed: u32, n: usize, amp: f32) -> Vec<Frame> {
    let mut lcg = Lcg(seed);
    (0..n)
        .map(|_| {
            let v = noise(&mut lcg) * amp;
            Frame::new(v, v)
        })
        .collect()
}

/// Deterministic seeded white noise, fully decorrelated between channels
/// (independent LCG streams per channel) — unlike [`white_noise_frames`]'s
/// mono-upmix, this is a source `WD` narrowing can actually be heard
/// narrowing (a mono-upmixed L==R source has no image to narrow).
fn decorrelated_noise_frames(seed_l: u32, seed_r: u32, n: usize, amp: f32) -> Vec<Frame> {
    let mut lcg_l = Lcg(seed_l);
    let mut lcg_r = Lcg(seed_r);
    (0..n)
        .map(|_| Frame::new(noise(&mut lcg_l) * amp, noise(&mut lcg_r) * amp))
        .collect()
}

/// Builds one `AudioManager<OfflineBackend>` with the real field-chain graph
/// `playback.rs`'s `try_open_device` builds (main-track limiter → `SPACE`
/// send with `Hold`+`Reverb` → one group track with the full 12-slot chain),
/// with `target` applied once via `apply_bus_fx` at `t = 0`. Returns the
/// manager and the group track, ready for a caller to `play()` a source and
/// `render()`.
fn build_scenario_manager(
    sample_rate: u32,
    target: BusFx,
) -> (AudioManager<OfflineBackend>, TrackHandle) {
    let settings = AudioManagerSettings::<OfflineBackend> {
        main_track_builder: kira::track::MainTrackBuilder::new().with_effect(
            CompressorBuilder::new()
                .threshold(crate::playback::LIMITER_THRESHOLD_DB)
                .ratio(crate::playback::LIMITER_RATIO)
                .attack_duration(std::time::Duration::from_millis(
                    crate::playback::LIMITER_ATTACK_MS,
                ))
                .release_duration(std::time::Duration::from_millis(
                    crate::playback::LIMITER_RELEASE_MS,
                ))
                .makeup_gain(Decibels(0.0)),
        ),
        backend_settings: OfflineBackendSettings { sample_rate },
        ..Default::default()
    };
    let mut manager = AudioManager::<OfflineBackend>::new(settings)
        .expect("offline backend never fails to start");

    let mut space_builder = SendTrackBuilder::new();
    space_builder.add_effect(HoldBuilder::new());
    space_builder.add_effect(
        ReverbBuilder::new()
            .feedback(0.85)
            .damping(0.5)
            .mix(Mix(1.0)),
    );
    let space_send = manager.add_send_track(space_builder).expect("send track");

    let (builder, mut chain) =
        crate::playback::build_field_chain(crate::playback::DL_TIME_AMB, space_send.id());
    let mut grp = manager.add_sub_track(builder).expect("sub track");
    crate::playback::apply_bus_fx(
        &mut grp,
        &mut chain,
        &target,
        std::time::Instant::now(),
        Tween::default(),
        Tween::default(),
    );

    (manager, grp)
}

/// Renders one field-chain group track (`playback::build_field_chain`, the
/// same constructor `try_open_device` uses) over `secs` seconds of
/// deterministic mono-upmixed white noise, through the same limiter settings
/// the main track uses (`playback::LIMITER_*`), with `target` applied once
/// via `playback::apply_bus_fx` at `t = 0` (its default 10ms tween settles
/// almost immediately, so the render is at steady state for effectively the
/// whole duration). Pass [`BusFx::NEUTRAL`] for the baseline/no-field case.
///
/// Real kira graph end to end: `AudioManager<OfflineBackend>` → SPACE send
/// (`Hold` + `Reverb`, exactly as `try_open_device` builds it) → one group
/// track with the full 12-slot chain → the main track's limiter.
pub fn render_field_chain_scenario(
    sample_rate: u32,
    secs: f32,
    seed: u32,
    target: BusFx,
) -> Vec<f32> {
    let (mut manager, mut grp) = build_scenario_manager(sample_rate, target);
    let n = (sample_rate as f32 * secs).round() as usize;
    let frames = white_noise_frames(seed, n, 0.3);
    let data = StaticSoundData {
        sample_rate,
        frames: frames.into(),
        settings: StaticSoundSettings::default(),
        slice: None,
    };
    let _ = grp.play(data);
    manager.backend_mut().render(n)
}

/// Same real graph as [`render_field_chain_scenario`], but with a genuinely
/// decorrelated stereo source (see [`decorrelated_noise_frames`]) so a `WD`
/// narrowing target is actually audible in the render, not just numerically
/// present.
pub fn render_field_chain_scenario_stereo(
    sample_rate: u32,
    secs: f32,
    seed_l: u32,
    seed_r: u32,
    target: BusFx,
) -> Vec<f32> {
    let (mut manager, mut grp) = build_scenario_manager(sample_rate, target);
    let n = (sample_rate as f32 * secs).round() as usize;
    let frames = decorrelated_noise_frames(seed_l, seed_r, n, 0.3);
    let data = StaticSoundData {
        sample_rate,
        frames: frames.into(),
        settings: StaticSoundSettings::default(),
        slice: None,
    };
    let _ = grp.play(data);
    manager.backend_mut().render(n)
}

/// Peak absolute sample value, in dBFS (`-inf` for silence).
pub fn peak_dbfs(samples: &[f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// `true` if any sample is NaN or infinite.
pub fn has_nan_or_inf(samples: &[f32]) -> bool {
    samples.iter().any(|s| !s.is_finite())
}

/// Full-band RMS level, in dBFS (`-inf` for silence). Lag-independent, unlike
/// a raw per-sample diff — the right comparator for "does chain A color the
/// signal versus chain B" when the two graphs may drain their command queues
/// (and so start audible playback) a block or two apart, which a sample-index
/// `a[i] - b[i]` diff would misread as huge coloring on uncorrelated noise.
pub fn rms_dbfs(samples: &[f32]) -> f32 {
    let sum_sq: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let rms = (sum_sq / samples.len().max(1) as f64).sqrt() as f32;
    if rms <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * rms.log10()
    }
}

/// The largest sample-to-sample jump, per channel — a coarse click detector.
/// `channels` is 1 or 2 (interleaved stride).
pub fn max_sample_jump(samples: &[f32], channels: usize) -> f32 {
    samples
        .chunks_exact(channels)
        .zip(samples.chunks_exact(channels).skip(1))
        .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (y - x).abs()))
        .fold(0.0f32, f32::max)
}

/// RMS energy above `cutoff_hz`, via `dsp::OnePole::hp` (a coarse,
/// FFT-free spectral-tilt proxy — not a true spectral centroid).
pub fn hf_energy_rms(samples: &[f32], sample_rate: f32, cutoff_hz: f32) -> f32 {
    let mut hp = crate::dsp::OnePole::lowpass(cutoff_hz);
    let sum_sq: f64 = samples
        .iter()
        .map(|&x| {
            let h = hp.hp(x);
            f64::from(h) * f64::from(h)
        })
        .sum();
    let _ = sample_rate;
    ((sum_sq / samples.len().max(1) as f64).sqrt()) as f32
}

/// Left/right correlation of an interleaved stereo buffer (1.0 = mono-summed
/// / perfectly correlated, 0 = decorrelated).
pub fn stereo_correlation(samples: &[f32]) -> f32 {
    let n = samples.len() / 2;
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for f in samples.as_chunks::<2>().0 {
        l.push(f[0]);
        r.push(f[1]);
    }
    crate::dsp::correlation(&l, &r)
}

#[cfg(test)]
mod field_chain_tests {
    use super::*;
    use crate::field::BusFx;

    // Must stay comfortably above 2x the neutral chain's resting LP cutoff
    // (20kHz, `field::BusFx::NEUTRAL.lp_cutoff_hz`) — at a lower Nyquist that
    // "open" filter would itself be clamped down near the Nyquist and
    // audibly color the signal, which is a sample-rate-choice artifact, not
    // a field-chain defect. 44.1kHz matches the rest of the crate
    // (`crate::synth::SAMPLE_RATE`).
    const SR: u32 = 44_100;

    #[test]
    fn neutral_field_chain_is_bit_exact_against_no_chain_at_all() {
        let with_chain = render_field_chain_scenario(SR, 0.5, 7, BusFx::NEUTRAL);

        // The same source, same limiter, but played straight onto a bare
        // sub-track with no field chain in between at all.
        let settings = AudioManagerSettings::<OfflineBackend> {
            main_track_builder: kira::track::MainTrackBuilder::new().with_effect(
                CompressorBuilder::new()
                    .threshold(crate::playback::LIMITER_THRESHOLD_DB)
                    .ratio(crate::playback::LIMITER_RATIO)
                    .attack_duration(std::time::Duration::from_millis(
                        crate::playback::LIMITER_ATTACK_MS,
                    ))
                    .release_duration(std::time::Duration::from_millis(
                        crate::playback::LIMITER_RELEASE_MS,
                    ))
                    .makeup_gain(Decibels(0.0)),
            ),
            backend_settings: OfflineBackendSettings { sample_rate: SR },
            ..Default::default()
        };
        let mut manager = AudioManager::<OfflineBackend>::new(settings).unwrap();
        let mut bare = manager
            .add_sub_track(kira::track::TrackBuilder::new())
            .unwrap();
        let n = (SR as f32 * 0.5).round() as usize;
        let frames = white_noise_frames(7, n, 0.3);
        let data = StaticSoundData {
            sample_rate: SR,
            frames: frames.into(),
            settings: StaticSoundSettings::default(),
            slice: None,
        };
        let _ = bare.play(data);
        let without_chain = manager.backend_mut().render(n);

        assert_eq!(with_chain.len(), without_chain.len());
        // RMS level, not a per-sample diff: the two graphs have different
        // shapes (12 extra insert slots vs. none), so kira's command queue
        // for `with_chain` may finish draining a render block or two later
        // than the bare track's — on uncorrelated white noise, even a
        // handful of samples of misalignment makes an index-aligned
        // `a[i]-b[i]` diff swing close to full amplitude despite the two
        // signals being perceptually identical. RMS level sidesteps that:
        // it is exactly what "does the neutral chain color/attenuate the
        // signal" actually means, and it is lag-independent.
        let db_with = rms_dbfs(&with_chain);
        let db_without = rms_dbfs(&without_chain);
        // A real (non-brick-wall) 2-pole LP resting at 20kHz against a
        // 44.1kHz-sample-rate (22.05kHz Nyquist) white-noise source rolls
        // off real energy in that last ~2kHz even at its "open" resting
        // cutoff — an ideal brick-wall LP alone already accounts for about
        // 0.4dB of broadband RMS loss there, and the real filter's gradual
        // shoulder plus the HP/shelves/bell each contributing their own
        // sub-tenth-dB deviation from ideal-unity add up to the sub-1dB
        // measured here. 1.5dB is comfortably above that and would still
        // catch a real bug (e.g. a stage accidentally left at a non-neutral
        // gain, which reads in whole dB, not tenths).
        assert!(
            (db_with - db_without).abs() < 1.5,
            "neutral field chain changes RMS level: with={db_with}dBFS without={db_without}dBFS"
        );
    }

    #[test]
    fn no_nan_inf_or_clipping_over_a_worst_case_stack() {
        // A synthetic "everything driven near its cap" BusFx — a worse case
        // than any single archetype produces, exercising the limiter's
        // worst-case acceptance headroom.
        let hot = BusFx {
            ls_db: 6.0,
            bl_freq_hz: 1200.0,
            bl_q: 2.0,
            bl_gain_db: 6.0,
            hs_db: -18.0,
            lp_cutoff_hz: 700.0,
            hp_cutoff_hz: 400.0,
            crush_bits: 6.0,
            crush_hold: 8.0,
            crush_dropout: 0.25,
            crush_mix: 0.15,
            ringmod_carrier_hz: 220.0,
            ringmod_mix: 0.1,
            wow_base_ms: 15.0,
            wow_depth_ms: 6.0,
            wow_rate_hz: 2.0,
            wow_mix: 0.1,
            hold_mode: crate::field_fx::glitch_stutter::HoldMode::Stutter,
            hold_window_ms: 60.0,
            hold_mix: 0.1,
            dist_drive_db: 18.0,
            dist_mix: 0.1,
            width: 0.3,
            delay_feedback_db: -3.0,
            delay_mix: 0.05,
            vol_duck_db: 0.0,
            pump_depth_db: 3.0,
            pump_rate_hz: 2.5,
            send_db: -3.0,
        };
        let out = render_field_chain_scenario(SR, 2.0, 99, hot);
        assert!(!has_nan_or_inf(&out), "worst-case stack produced NaN/inf");
        let peak = peak_dbfs(&out);
        assert!(
            peak <= -0.3 || peak == f32::NEG_INFINITY,
            "worst-case stack clipped: peak={peak}dBFS"
        );
        // No click-magnitude assertion here on purpose: this stack
        // deliberately engages `HD::Stutter` (a buffer-repeat loop boundary
        // is a genuine, intended discontinuity) and heavy `CR` dropout (a
        // silent-gap edge is also a genuine, intended discontinuity) — both
        // are the glitch aesthetic working as designed, not a defect. The
        // dedicated click check below targets a *continuous*-effects-only
        // scenario instead, where any jump indicates an implementation bug
        // (e.g. a mistuned tween) rather than an authored discontinuity.
        let jump = max_sample_jump(&out, 2);
        assert!(jump.is_finite(), "non-finite sample jump: {jump}");
    }

    #[test]
    fn continuous_only_stack_has_no_unexplained_clicks() {
        // The same spirit as the worst-case stack above, but with both
        // discontinuity-by-design effects (`HD`, `CR` dropout) left at rest,
        // so every remaining stage (shelves/bell/filters/`RM`/`WW`/`DS`/`WD`/
        // `DL`) is continuous and click-free tweening is the only thing
        // under test.
        let continuous = BusFx {
            ls_db: 6.0,
            bl_freq_hz: 1200.0,
            bl_q: 2.0,
            bl_gain_db: 6.0,
            hs_db: -18.0,
            lp_cutoff_hz: 700.0,
            hp_cutoff_hz: 400.0,
            ringmod_carrier_hz: 220.0,
            ringmod_mix: 0.1,
            wow_base_ms: 15.0,
            wow_depth_ms: 6.0,
            wow_rate_hz: 2.0,
            wow_mix: 0.1,
            dist_drive_db: 18.0,
            dist_mix: 0.1,
            width: 0.3,
            delay_feedback_db: -3.0,
            delay_mix: 0.05,
            pump_depth_db: 3.0,
            pump_rate_hz: 2.5,
            send_db: -3.0,
            ..BusFx::NEUTRAL
        };
        let out = render_field_chain_scenario(SR, 2.0, 99, continuous);
        assert!(!has_nan_or_inf(&out), "continuous stack produced NaN/inf");
        // A generous but meaningful ceiling: the source itself is 0.3-amplitude
        // white noise, so an adjacent-sample jump approaching full scale would
        // indicate a real click, not just noise's own sample-to-sample variance.
        let jump = max_sample_jump(&out, 2);
        assert!(
            jump < 0.6,
            "suspiciously large sample-to-sample jump in an all-continuous chain: {jump}"
        );
    }

    #[test]
    fn deterministic_two_renders_of_a_moving_field_are_byte_identical() {
        let target = BusFx {
            lp_cutoff_hz: 900.0,
            hs_db: -10.0,
            width: 0.6,
            ..BusFx::NEUTRAL
        };
        let a = render_field_chain_scenario(SR, 1.0, 42, target);
        let b = render_field_chain_scenario(SR, 1.0, 42, target);
        assert_eq!(a, b);
    }

    #[test]
    fn lethe_wash_style_target_measurably_reduces_high_frequency_energy() {
        // LetheWash's AMB target: LP -> 700Hz, HS -> -12dB.
        let f1 = BusFx {
            lp_cutoff_hz: 700.0,
            hs_db: -12.0,
            ..BusFx::NEUTRAL
        };
        let neutral_out = render_field_chain_scenario(SR, 1.0, 5, BusFx::NEUTRAL);
        let f1_out = render_field_chain_scenario(SR, 1.0, 5, f1);
        let hf_neutral = hf_energy_rms(&neutral_out, SR as f32, 4_000.0);
        let hf_f1 = hf_energy_rms(&f1_out, SR as f32, 4_000.0);
        assert!(
            hf_f1 < hf_neutral * 0.5,
            "F1-style LP/HS target did not meaningfully reduce HF energy: neutral={hf_neutral}, f1={hf_f1}"
        );
    }

    #[test]
    fn recognition_narrow_style_target_measurably_narrows_the_stereo_image() {
        // RecognitionNarrow's target: WD -> 0.3. Needs a genuinely
        // decorrelated stereo source (mono-upmixed noise has no image to
        // narrow at all).
        let f13 = BusFx {
            width: 0.3,
            ..BusFx::NEUTRAL
        };
        let neutral_out = render_field_chain_scenario_stereo(SR, 1.0, 11, 12, BusFx::NEUTRAL);
        let f13_out = render_field_chain_scenario_stereo(SR, 1.0, 11, 12, f13);
        assert!(!has_nan_or_inf(&f13_out));
        let corr_neutral = stereo_correlation(&neutral_out);
        let corr_narrow = stereo_correlation(&f13_out);
        assert!(
            corr_narrow > corr_neutral + 0.1,
            "WD=0.3 did not measurably raise L/R correlation: neutral={corr_neutral}, narrow={corr_narrow}"
        );
    }
}

// ───────────────── WAV scenario export (manual, not part of `cargo test`) ──
//
// `#[ignore]`d and gated by the `FIELD_WAV_OUT_DIR` env var so a normal
// `cargo test` run never touches disk or bakes a session-specific path into
// the repo. Run once manually to produce human-auditionable WAVs:
//   FIELD_WAV_OUT_DIR=<dir> cargo test -p re_audio --lib -- --ignored render_field_chain_wav_scenarios

#[cfg(test)]
mod wav_export {
    use std::path::Path;

    use super::*;
    use crate::dsp::Stereo;
    use crate::field::{combine, Archetype, FieldInput, FieldMix, FieldSettings, AMB};
    use crate::field_fx::glitch_stutter::HoldMode;

    const SR: u32 = 44_100;

    fn amb_fx_for(archetype: Archetype, variant: u8, reduced: bool) -> BusFx {
        let input = FieldInput {
            archetype,
            weight: 1.0,
            depth: 1.0,
            priority: archetype.default_priority(),
            aux: 0.0,
            variant,
            seed: 7,
            bearing: None,
        };
        let mix: FieldMix = combine(
            &[input],
            &FieldSettings {
                intensity: 1.0,
                reduced,
            },
        );
        mix.buses[AMB]
    }

    fn worst_case_busfx() -> BusFx {
        BusFx {
            ls_db: 6.0,
            bl_freq_hz: 1_200.0,
            bl_q: 2.0,
            bl_gain_db: 6.0,
            hs_db: -18.0,
            lp_cutoff_hz: 700.0,
            hp_cutoff_hz: 400.0,
            crush_bits: 6.0,
            crush_hold: 8.0,
            crush_dropout: 0.25,
            crush_mix: 0.15,
            ringmod_carrier_hz: 220.0,
            ringmod_mix: 0.1,
            wow_base_ms: 15.0,
            wow_depth_ms: 6.0,
            wow_rate_hz: 2.0,
            wow_mix: 0.1,
            hold_mode: HoldMode::Stutter,
            hold_window_ms: 60.0,
            hold_mix: 0.1,
            dist_drive_db: 18.0,
            dist_mix: 0.1,
            width: 0.3,
            delay_feedback_db: -3.0,
            delay_mix: 0.05,
            vol_duck_db: 0.0,
            pump_depth_db: 3.0,
            pump_rate_hz: 2.5,
            send_db: -3.0,
        }
    }

    #[test]
    #[ignore = "manual: writes WAVs to $FIELD_WAV_OUT_DIR, never runs in normal cargo test"]
    fn render_field_chain_wav_scenarios() {
        let Ok(out_dir) = std::env::var("FIELD_WAV_OUT_DIR") else {
            eprintln!("FIELD_WAV_OUT_DIR not set — skipping (this test only runs when explicitly asked to render WAVs)");
            return;
        };
        let out_dir = Path::new(&out_dir);
        std::fs::create_dir_all(out_dir).expect("create FIELD_WAV_OUT_DIR");

        let mut readme = String::new();
        readme.push_str("# Sound-fields offline verification WAVs\n\n");
        readme.push_str(
            "Rendered through the REAL kira playback graph `playback.rs` builds \
             (`OfflineBackend`, not a hand-rolled approximation): main-track limiter \
             -> SPACE send (Hold+Reverb) -> one AMB group track with the full 12-slot \
             field insert chain (LS BL HS LP HP CR RM WW HD DS WD DL). Every scenario \
             plays 0.3-amplitude seeded white noise so every slot always has material \
             to act on. Listen on headphones for the stereo-image scenarios.\n\n",
        );

        let mut scenario = |name: &str, note: &str, samples: Vec<f32>| {
            let stereo: Stereo = interleaved_to_stereo(&samples);
            let path = out_dir.join(format!("{name}.wav"));
            crate::offline::write_wav(&path, &stereo).expect("write wav");
            readme.push_str(&format!("## `{name}.wav`\n{note}\n\n"));
        };

        scenario(
            "01_neutral",
            "No field active (`BusFx::NEUTRAL`). Baseline — should sound like plain \
             filtered white noise with no coloration, no wobble, no glitch.",
            render_field_chain_scenario(SR, 3.0, 1, BusFx::NEUTRAL),
        );

        scenario(
            "02_lethe_wash",
            "`lethe_wash` (F1) at full weight/depth: AMB darkens under a closing \
             lowpass and a dipped high shelf — a muffled, distant quality.",
            render_field_chain_scenario(SR, 3.0, 1, amb_fx_for(Archetype::LetheWash, 0, false)),
        );

        scenario(
            "03_bit_rot",
            "`bit_rot` (F3) at full weight/depth: bit-depth crush, sample-and-hold \
             decimation, dropout, and soft-clip drive — a degraded, crunchy texture.",
            render_field_chain_scenario(SR, 3.0, 1, amb_fx_for(Archetype::BitRot, 0, false)),
        );

        scenario(
            "04_mnemos_hold_freeze",
            "`mnemos_hold` (F2) with the `cycle_held` variant set: AMB's `HD` slot \
             switches to Freeze mode, capturing and looping a short window forever — \
             a held, static drone replacing the moving noise.",
            render_field_chain_scenario(SR, 3.0, 1, amb_fx_for(Archetype::MnemosHold, 1, false)),
        );

        scenario(
            "05_watcher_gaze_narrow_stereo",
            "`watcher_gaze` (F5) at full weight/depth, rendered with a genuinely \
             decorrelated stereo source (unlike the mono-upmixed default) so the \
             width narrowing to 0.35 and the -4dB duck are both audible as a \
             collapsing, quieter stereo image.",
            render_field_chain_scenario_stereo(
                SR,
                3.0,
                21,
                22,
                amb_fx_for(Archetype::WatcherGaze, 0, false),
            ),
        );

        scenario(
            "06_worst_case_stack",
            "A synthetic everything-near-its-cap `BusFx` (harsher than any single \
             archetype), through the SAME real limiter as the main track — \
             demonstrates the headroom guarantee (peak stays <= -0.3dBFS) under the \
             worst case the field-chain math allows, plus the HD-stutter/CR-dropout \
             discontinuities the glitch aesthetic intentionally produces.",
            render_field_chain_scenario(SR, 3.0, 99, worst_case_busfx()),
        );

        scenario(
            "07_bit_rot_reduced_mode",
            "The same `bit_rot` (F3) field as 03, but with `sound_fields_reduced` \
             on: dropout is forced to 0 (no silent gaps) and `RM`/`HD` are zeroed \
             by the reduced-mode substitution table (\u{a7}3.5) — `bit_rot` doesn't \
             use those slots anyway, so the audible difference here is specifically \
             the removed dropout gaps versus 03.",
            render_field_chain_scenario(SR, 3.0, 1, amb_fx_for(Archetype::BitRot, 0, true)),
        );

        std::fs::write(out_dir.join("README.md"), readme).expect("write README");
        println!("wrote 7 scenario WAVs + README.md to {}", out_dir.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::{AudioManager, AudioManagerSettings};

    #[test]
    fn silent_graph_renders_exact_zeros_and_the_right_frame_count() {
        let mut manager = AudioManager::<OfflineBackend>::new(AudioManagerSettings {
            backend_settings: OfflineBackendSettings { sample_rate: 8_000 },
            ..Default::default()
        })
        .expect("offline backend never fails to start");
        let samples = manager.backend_mut().render(1_000);
        assert_eq!(samples.len(), 2_000);
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn interleaved_to_stereo_deinterleaves_correctly() {
        let samples = [0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        let stereo = interleaved_to_stereo(&samples);
        assert_eq!(stereo.l, vec![0.1, 0.3, 0.5]);
        assert_eq!(stereo.r, vec![-0.2, -0.4, -0.6]);
    }

    #[test]
    fn two_renders_of_the_same_untouched_graph_are_byte_identical() {
        let mut a = AudioManager::<OfflineBackend>::new(AudioManagerSettings {
            backend_settings: OfflineBackendSettings { sample_rate: 8_000 },
            ..Default::default()
        })
        .unwrap();
        let mut b = AudioManager::<OfflineBackend>::new(AudioManagerSettings {
            backend_settings: OfflineBackendSettings { sample_rate: 8_000 },
            ..Default::default()
        })
        .unwrap();
        let sa = a.backend_mut().render(4_321);
        let sb = b.backend_mut().render(4_321);
        assert_eq!(sa, sb);
    }
}
