//! `dsp` — small, deterministic signal-processing primitives. **Pure maths, no device.**
//!
//! The building blocks the stereo [`crate::soundscape`] and the room model in
//! [`crate::acoustics`] are made of: one-pole filters, integer delay lines, a
//! Freeverb-style comb/all-pass pair, an equal-power pan law, a seamless-loop
//! crossfade, and a linear resampler (what kira's `playback_rate` does, for the
//! offline mixdown). Everything is `f32`, allocation happens only at
//! construction, and no function reads a clock or an unseeded RNG — the same
//! inputs always produce the same samples.

use std::f32::consts::FRAC_PI_2;

use crate::synth::{noise, Lcg, SAMPLE_RATE};

/// A stereo PCM buffer at [`SAMPLE_RATE`]: two equal-length channels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stereo {
    /// Left channel samples.
    pub l: Vec<f32>,
    /// Right channel samples.
    pub r: Vec<f32>,
}

impl Stereo {
    /// Two silent channels of `n` frames.
    pub fn silent(n: usize) -> Stereo {
        Stereo {
            l: vec![0.0; n],
            r: vec![0.0; n],
        }
    }

    /// Duplicate a mono buffer into both channels.
    pub fn from_mono(m: &[f32]) -> Stereo {
        Stereo {
            l: m.to_vec(),
            r: m.to_vec(),
        }
    }

    /// Frame count (both channels are always the same length).
    #[inline]
    pub fn len(&self) -> usize {
        self.l.len()
    }

    /// `true` for a zero-length buffer.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.l.is_empty()
    }

    /// Multiply both channels by `g`.
    pub fn scale(&mut self, g: f32) {
        for v in self.l.iter_mut().chain(self.r.iter_mut()) {
            *v *= g;
        }
    }

    /// Peak absolute sample over both channels.
    pub fn peak(&self) -> f32 {
        self.l
            .iter()
            .chain(self.r.iter())
            .fold(0.0f32, |m, v| m.max(v.abs()))
    }

    /// Scale so the peak sits at `target` (no-op for silence).
    pub fn normalize_to(&mut self, target: f32) {
        let p = self.peak();
        if p > 1e-9 {
            self.scale(target / p);
        }
    }

    /// RMS over both channels.
    pub fn rms(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let s: f64 = self
            .l
            .iter()
            .chain(self.r.iter())
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        (s / (2 * self.len()) as f64).sqrt() as f32
    }

    /// Pearson correlation between the two channels in `[-1, 1]`: `1` = mono
    /// (identical), `≈0` = fully decorrelated (the widest stereo image).
    pub fn lr_correlation(&self) -> f32 {
        correlation(&self.l, &self.r)
    }

    /// Clamp every sample into `[-1, 1]`.
    pub fn clamp_unit(&mut self) {
        for v in self.l.iter_mut().chain(self.r.iter_mut()) {
            *v = v.clamp(-1.0, 1.0);
        }
    }
}

/// Pearson correlation of two equal-length buffers (0 if either is silent).
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let ma = a[..n].iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let mb = b[..n].iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let da = f64::from(a[i]) - ma;
        let db = f64::from(b[i]) - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 1e-18 || sbb <= 1e-18 {
        return 0.0;
    }
    (sab / (saa.sqrt() * sbb.sqrt())) as f32
}

/// Equal-power pan law: `pan` in `[-1, 1]` (−1 hard left) → `(gain_l, gain_r)`
/// with `gl² + gr² == 1`, so a source keeps its loudness as it moves.
#[inline]
pub fn pan_gains(pan: f32) -> (f32, f32) {
    let p = (pan.clamp(-1.0, 1.0) + 1.0) * 0.5; // 0..1
    let a = p * FRAC_PI_2;
    (a.cos(), a.sin())
}

/// One-pole low-pass: `y += k·(x − y)` with `k` from a cutoff in Hz.
#[derive(Clone, Copy, Debug, Default)]
pub struct OnePole {
    k: f32,
    y: f32,
}

impl OnePole {
    /// A low-pass at `cutoff_hz` (clamped to a sane audio range).
    pub fn lowpass(cutoff_hz: f32) -> OnePole {
        let fc = cutoff_hz.clamp(5.0, SAMPLE_RATE as f32 * 0.45);
        let k = 1.0 - (-2.0 * std::f32::consts::PI * fc / SAMPLE_RATE as f32).exp();
        OnePole { k, y: 0.0 }
    }

    /// Filter one sample (low-pass output).
    #[inline]
    pub fn lp(&mut self, x: f32) -> f32 {
        self.y += self.k * (x - self.y);
        self.y
    }

    /// Change the cutoff live, without resetting the filter's history. Used
    /// to sweep a filter slowly under an LFO (the drone music's filter
    /// movement) or to retarget it per-frame to a spatial cutoff (air
    /// absorption / occlusion) — cheaper than rebuilding the filter, and the
    /// in-flight sample keeps its continuity (no click).
    #[inline]
    pub fn retune(&mut self, cutoff_hz: f32) {
        let fc = cutoff_hz.clamp(5.0, SAMPLE_RATE as f32 * 0.45);
        self.k = 1.0 - (-2.0 * std::f32::consts::PI * fc / SAMPLE_RATE as f32).exp();
    }

    /// Filter one sample and return the complementary high-pass (`x − lp(x)`).
    #[inline]
    pub fn hp(&mut self, x: f32) -> f32 {
        x - self.lp(x)
    }
}

/// A fixed integer delay line (ring buffer).
#[derive(Clone, Debug)]
pub struct Delay {
    buf: Vec<f32>,
    pos: usize,
}

impl Delay {
    /// A delay of `samples` (≥ 1).
    pub fn new(samples: usize) -> Delay {
        Delay {
            buf: vec![0.0; samples.max(1)],
            pos: 0,
        }
    }

    /// Push `x`, return the sample written `len` pushes ago.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let out = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        out
    }
}

/// A short ring buffer supporting a read at a variable, small delay — unlike
/// [`Delay`] (one fixed delay baked in at construction), the read offset can
/// change from sample to sample. Sized for interaural time difference (at
/// most a fraction of a millisecond — a handful of samples), not for a
/// reverb-scale delay line.
#[derive(Clone, Debug)]
pub struct VariableDelay {
    buf: Vec<f32>,
    pos: usize,
}

impl VariableDelay {
    /// A buffer holding the last `capacity` pushed samples (`≥ 1`).
    pub fn new(capacity: usize) -> VariableDelay {
        VariableDelay {
            buf: vec![0.0; capacity.max(1)],
            pos: 0,
        }
    }

    /// Push the newest sample.
    #[inline]
    pub fn push(&mut self, x: f32) {
        self.buf[self.pos] = x;
        self.pos = (self.pos + 1) % self.buf.len();
    }

    /// Read the sample `delay_samples` pushes ago (0 = the one just pushed).
    /// Clamped to the buffer's capacity.
    #[inline]
    pub fn read(&self, delay_samples: usize) -> f32 {
        let cap = self.buf.len();
        let d = delay_samples.min(cap - 1);
        let idx = (self.pos + cap - 1 - d) % cap;
        self.buf[idx]
    }
}

/// Freeverb-style lowpass-feedback comb filter.
#[derive(Clone, Debug)]
pub struct Comb {
    buf: Vec<f32>,
    pos: usize,
    feedback: f32,
    damp: f32,
    store: f32,
}

impl Comb {
    /// A comb of `samples` delay, loop `feedback` in `[0, 1)` and high-frequency
    /// `damp` in `[0, 1)` (0 = bright, → 1 = dark tail).
    pub fn new(samples: usize, feedback: f32, damp: f32) -> Comb {
        Comb {
            buf: vec![0.0; samples.max(1)],
            pos: 0,
            feedback: feedback.clamp(0.0, 0.995),
            damp: damp.clamp(0.0, 0.99),
            store: 0.0,
        }
    }

    /// Push one input sample, return the delayed+damped-feedback output.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let out = self.buf[self.pos];
        self.store = out * (1.0 - self.damp) + self.store * self.damp;
        self.buf[self.pos] = x + self.store * self.feedback;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        out
    }
}

/// Schroeder all-pass diffuser.
#[derive(Clone, Debug)]
pub struct Allpass {
    buf: Vec<f32>,
    pos: usize,
    g: f32,
}

impl Allpass {
    /// An all-pass diffuser of `samples` delay and coefficient `g` (typically
    /// `~0.5`; flat frequency response, only the phase/diffusion changes).
    pub fn new(samples: usize, g: f32) -> Allpass {
        Allpass {
            buf: vec![0.0; samples.max(1)],
            pos: 0,
            g,
        }
    }

    /// Push one input sample, return the diffused output.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let b = self.buf[self.pos];
        let out = b - x;
        self.buf[self.pos] = x + b * self.g;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        out
    }
}

/// Make a buffer loop seamlessly. `buf` must be `loop_len + xfade` long: the
/// trailing `xfade` samples are equal-power crossfaded into the head, and the
/// result is exactly `loop_len` samples whose last sample flows into its first.
/// Returns the buffer untouched (truncated) if it is too short to fade.
pub fn make_seamless(buf: &[f32], loop_len: usize, xfade: usize) -> Vec<f32> {
    if buf.len() < loop_len + xfade || xfade == 0 || xfade > loop_len {
        return buf[..loop_len.min(buf.len())].to_vec();
    }
    let mut out = buf[..loop_len].to_vec();
    for i in 0..xfade {
        let t = i as f32 / xfade as f32;
        let a = t * FRAC_PI_2;
        // Head fades in, the continuation past the loop point fades out.
        out[i] = buf[i] * a.sin() + buf[loop_len + i] * a.cos();
    }
    out
}

/// Deterministic smoothed random walk in `[0, 1]`: a value that drifts slowly
/// (new target every `period` samples, one-pole smoothed). Used for the slow,
/// never-repeating evolution of the ambience layers.
#[derive(Clone, Debug)]
pub struct Drift {
    lcg: Lcg,
    period: usize,
    count: usize,
    target: f32,
    smooth: OnePole,
}

impl Drift {
    /// A drift that picks a new target every `period_secs` and glides to it.
    pub fn new(seed: u32, period_secs: f32) -> Drift {
        let period = ((period_secs.max(0.01)) * SAMPLE_RATE as f32) as usize;
        let mut lcg = Lcg(seed);
        let target = noise(&mut lcg) * 0.5 + 0.5;
        let mut smooth = OnePole::lowpass(1.0 / period_secs.max(0.05));
        // Start settled on the first target, so there is no fade-in ramp.
        for _ in 0..2048 {
            smooth.lp(target);
        }
        Drift {
            lcg,
            period: period.max(1),
            count: 0,
            target,
            smooth,
        }
    }

    /// Advance one sample and return the current smoothed value in `[0, 1]`.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        self.count += 1;
        if self.count >= self.period {
            self.count = 0;
            self.target = noise(&mut self.lcg) * 0.5 + 0.5;
        }
        self.smooth.lp(self.target).clamp(0.0, 1.0)
    }
}

/// Linear-interpolation resample by `rate` (2.0 = an octave up, half as long) —
/// the offline stand-in for kira's `playback_rate`.
pub fn resample(src: &[f32], rate: f64) -> Vec<f32> {
    if src.is_empty() || rate <= 0.0 {
        return Vec::new();
    }
    if (rate - 1.0).abs() < 1e-9 {
        return src.to_vec();
    }
    let out_len = ((src.len() as f64) / rate).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * rate;
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let a = src[i0.min(src.len() - 1)];
        let b = src[(i0 + 1).min(src.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Seconds → whole samples at [`SAMPLE_RATE`].
#[inline]
pub fn secs(s: f32) -> usize {
    (s.max(0.0) * SAMPLE_RATE as f32).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pan_law_is_equal_power_and_ordered() {
        for p in [-1.0f32, -0.5, 0.0, 0.3, 1.0] {
            let (l, r) = pan_gains(p);
            assert!(
                (l * l + r * r - 1.0).abs() < 1e-5,
                "pan {p} not equal-power"
            );
        }
        let (l, r) = pan_gains(-1.0);
        assert!(l > 0.999 && r < 1e-3);
        let (l, r) = pan_gains(1.0);
        assert!(r > 0.999 && l < 1e-3);
        let (l, r) = pan_gains(0.0);
        assert!((l - r).abs() < 1e-6);
    }

    #[test]
    fn seamless_loop_has_no_seam_jump() {
        // A sine whose period does NOT divide the loop: a naive cut would jump.
        let n = 10_000;
        let xf = 2_000;
        let src: Vec<f32> = (0..n + xf).map(|i| (i as f32 * 0.0123).sin()).collect();
        let naive_jump = (src[n - 1] - src[0]).abs();
        let out = make_seamless(&src, n, xf);
        assert_eq!(out.len(), n);
        let seam = (out[n - 1] - out[0]).abs();
        let max_step = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            seam <= max_step * 1.01,
            "seam jump {seam} vs max step {max_step}"
        );
        assert!(naive_jump > seam, "test premise: the naive cut must jump");
    }

    #[test]
    fn correlation_extremes() {
        let a: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let neg: Vec<f32> = a.iter().map(|v| -v).collect();
        assert!((correlation(&a, &a) - 1.0).abs() < 1e-4);
        assert!((correlation(&a, &neg) + 1.0).abs() < 1e-4);
        assert_eq!(correlation(&a, &vec![0.0; 1000]), 0.0);
    }

    #[test]
    fn resample_changes_length_and_is_identity_at_one() {
        let s: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(resample(&s, 1.0), s);
        assert_eq!(resample(&s, 2.0).len(), 500);
        assert!(resample(&s, 0.5).len() >= 1999);
    }

    #[test]
    fn drift_is_deterministic_and_bounded() {
        let mut a = Drift::new(7, 0.5);
        let mut b = Drift::new(7, 0.5);
        for _ in 0..50_000 {
            let (x, y) = (a.tick(), b.tick());
            assert_eq!(x, y);
            assert!((0.0..=1.0).contains(&x));
        }
    }

    #[test]
    fn variable_delay_reads_back_the_pushed_history_at_any_offset() {
        let mut d = VariableDelay::new(8);
        for i in 0..8 {
            d.push(i as f32);
        }
        // Just pushed 7.0; delay 0 is the newest, delay 7 is the oldest still held.
        assert_eq!(d.read(0), 7.0);
        assert_eq!(d.read(1), 6.0);
        assert_eq!(d.read(7), 0.0);
        // Beyond capacity clamps to the oldest available sample.
        assert_eq!(d.read(99), d.read(7));
        // Continuing to push evicts the oldest.
        d.push(8.0);
        assert_eq!(d.read(0), 8.0);
        assert_eq!(d.read(7), 1.0);
    }

    #[test]
    fn onepole_retune_changes_cutoff_without_resetting_state() {
        let mut lp = OnePole::lowpass(200.0);
        let hi_before: Vec<f32> = (0..2000)
            .map(|i| lp.lp((i as f32 * 2.0 * std::f32::consts::PI * 5000.0 / 44_100.0).sin()))
            .collect();
        let settled = hi_before[1500..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            settled < 0.1,
            "200 Hz cutoff should already block 5 kHz: {settled}"
        );
        // Retune wide open: the SAME signal should now pass through near full
        // amplitude, without a discontinuity from resetting `y`.
        lp.retune(18_000.0);
        let hi_after: Vec<f32> = (0..2000)
            .map(|i| lp.lp((i as f32 * 2.0 * std::f32::consts::PI * 5000.0 / 44_100.0).sin()))
            .collect();
        let opened = hi_after[1500..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(opened > 0.8, "retuned open should pass 5 kHz: {opened}");
    }

    #[test]
    fn onepole_lowpass_attenuates_highs() {
        let mut lp = OnePole::lowpass(200.0);
        let hi: Vec<f32> = (0..4410)
            .map(|i| lp.lp((i as f32 * 2.0 * std::f32::consts::PI * 5000.0 / 44_100.0).sin()))
            .collect();
        let peak = hi[2000..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.1, "5 kHz leaked through a 200 Hz low-pass: {peak}");
    }
}
