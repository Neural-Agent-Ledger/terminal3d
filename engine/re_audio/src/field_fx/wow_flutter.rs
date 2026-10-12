//! Tape wow & flutter: a continuously-modulated short delay (a combined slow
//! "wow" + fast "flutter" LFO drives a drifting fractional read lag around a
//! fixed base delay), which reads as pitch wobble. Kira's
//! [`DelayBuilder`](kira::effect::delay::DelayBuilder) fixes `delay_time` at
//! construction with no live tween and no fractional or modulated read —
//! there is nothing to reuse (see the module doc on [`super`]).
//!
//! Feeds the field chain's `WW` slot: `base` (3-20 ms, a fixed delay), `depth`
//! (0-6 ms, the peak deviation from `base`), `rate` (0.1-3 Hz), `mix` (0-1).
//! At `depth = 0`, `Wow` is a pure delay of `base` ms — depth does NOT
//! collapse the delay to zero; it is the modulation amplitude *around* the
//! fixed base.

use kira::command::ValueChangeCommand;
use kira::effect::{Effect, EffectBuilder};
use kira::info::Info;
use kira::{Frame, Mix, Parameter, Tween, Value};

use crate::synth::{noise, Lcg};

/// The largest `base_ms + depth_ms` the internal ring buffer is sized for; a
/// larger request is clamped. Comfortably covers the field chain's `base`
/// (<=20ms) + `depth` (<=6ms) range with headroom.
const MAX_TOTAL_MS: f64 = 30.0;

/// Configures a [`WowFlutter`] effect.
///
/// At `depth_ms = 0.0` the modulated lag collapses to exactly `base_ms`
/// (a pure, unmodulated delay) — not to zero. The only bit-exact-identity
/// bypass is `mix = 0.0` (the default), matching every other field effect's
/// dry/wet convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WowFlutterBuilder {
    /// Fixed base delay in milliseconds (`[0, 20]`; the field chain's `WW`
    /// slot uses `[3, 20]`).
    pub base_ms: Value<f64>,
    /// Peak modulation deviation from `base_ms`, in milliseconds (the field
    /// chain's `WW` slot uses `[0, 6]`; the ring buffer supports up to
    /// `MAX_TOTAL_MS - base_ms`).
    pub depth_ms: Value<f64>,
    /// Slow drift rate in Hz (typically well under 2 Hz).
    pub wow_rate_hz: Value<f64>,
    /// Fast flutter rate in Hz (typically 4–12 Hz).
    pub flutter_rate_hz: Value<f64>,
    /// Dry/wet blend. `0.0` (the default) is bit-exact passthrough.
    pub mix: Value<Mix>,
    /// Deterministic seed for the two LFOs' initial phases.
    pub seed: u32,
}

impl WowFlutterBuilder {
    /// A builder with `depth_ms` at its neutral default of `0.0` (pure fixed delay).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the fixed base delay, in milliseconds (see the `base_ms` field doc).
    #[must_use = "This method consumes self and returns a modified WowFlutterBuilder, so the return value should be used"]
    pub fn base_ms(self, base_ms: impl Into<Value<f64>>) -> Self {
        Self {
            base_ms: base_ms.into(),
            ..self
        }
    }

    /// Sets the peak modulation deviation from `base_ms` (see the `depth_ms`
    /// field doc).
    #[must_use = "This method consumes self and returns a modified WowFlutterBuilder, so the return value should be used"]
    pub fn depth_ms(self, depth_ms: impl Into<Value<f64>>) -> Self {
        Self {
            depth_ms: depth_ms.into(),
            ..self
        }
    }

    /// Sets the slow "wow" drift rate, in Hz (see the `wow_rate_hz` field doc).
    #[must_use = "This method consumes self and returns a modified WowFlutterBuilder, so the return value should be used"]
    pub fn wow_rate_hz(self, wow_rate_hz: impl Into<Value<f64>>) -> Self {
        Self {
            wow_rate_hz: wow_rate_hz.into(),
            ..self
        }
    }

    /// Sets the fast "flutter" rate, in Hz (see the `flutter_rate_hz` field doc).
    #[must_use = "This method consumes self and returns a modified WowFlutterBuilder, so the return value should be used"]
    pub fn flutter_rate_hz(self, flutter_rate_hz: impl Into<Value<f64>>) -> Self {
        Self {
            flutter_rate_hz: flutter_rate_hz.into(),
            ..self
        }
    }

    /// Sets the dry/wet blend (see the `mix` field doc).
    #[must_use = "This method consumes self and returns a modified WowFlutterBuilder, so the return value should be used"]
    pub fn mix(self, mix: impl Into<Value<Mix>>) -> Self {
        Self {
            mix: mix.into(),
            ..self
        }
    }

    /// Sets the deterministic seed for the two LFOs' initial phases.
    #[must_use = "This method consumes self and returns a modified WowFlutterBuilder, so the return value should be used"]
    pub fn seed(self, seed: u32) -> Self {
        Self { seed, ..self }
    }
}

impl Default for WowFlutterBuilder {
    fn default() -> Self {
        Self {
            base_ms: Value::Fixed(8.0),
            depth_ms: Value::Fixed(0.0),
            wow_rate_hz: Value::Fixed(0.6),
            flutter_rate_hz: Value::Fixed(7.0),
            mix: Value::Fixed(Mix::DRY),
            seed: 0,
        }
    }
}

impl EffectBuilder for WowFlutterBuilder {
    type Handle = WowFlutterHandle;

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        let (command_writers, command_readers) = command_writers_and_readers();
        let mut lcg = Lcg(self.seed ^ 0x0B0B_5EED);
        let wow_phase = (f64::from(noise(&mut lcg)) * 0.5 + 0.5) * std::f64::consts::TAU;
        let flutter_phase = (f64::from(noise(&mut lcg)) * 0.5 + 0.5) * std::f64::consts::TAU;
        (
            Box::new(WowFlutter {
                command_readers,
                base_ms: Parameter::new(self.base_ms, 8.0),
                depth_ms: Parameter::new(self.depth_ms, 0.0),
                wow_rate_hz: Parameter::new(self.wow_rate_hz, 0.6),
                flutter_rate_hz: Parameter::new(self.flutter_rate_hz, 7.0),
                mix: Parameter::new(self.mix, Mix::DRY),
                wow_phase,
                flutter_phase,
                sample_rate: 44_100.0,
                buf_l: Vec::new(),
                buf_r: Vec::new(),
                pos: 0,
            }),
            WowFlutterHandle { command_writers },
        )
    }
}

struct WowFlutter {
    command_readers: CommandReaders,
    base_ms: Parameter<f64>,
    depth_ms: Parameter<f64>,
    wow_rate_hz: Parameter<f64>,
    flutter_rate_hz: Parameter<f64>,
    mix: Parameter<Mix>,
    wow_phase: f64,
    flutter_phase: f64,
    sample_rate: f64,
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    /// Ring buffer write position (index of the *next* slot to write).
    pos: usize,
}

/// Read the ring buffer `delay` samples (fractional, `>= 0`) back from the
/// slot just before `pos` (the next slot to be written). `delay = 0.0` reads
/// back exactly the most recently pushed sample — bit-exact, no interpolation
/// error, since the fractional weight on the second tap is then `0.0`.
#[inline]
fn fractional_read(buf: &[f32], pos: usize, delay: f64) -> f32 {
    let cap = buf.len();
    let d0 = (delay.floor() as usize).min(cap - 1);
    let d1 = (d0 + 1).min(cap - 1);
    let frac = (delay - d0 as f64) as f32;
    let a = buf[(pos + cap - 1 - d0) % cap];
    let b = buf[(pos + cap - 1 - d1) % cap];
    a + (b - a) * frac
}

impl WowFlutter {
    fn resize(&mut self, sample_rate: u32) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
        let cap = ((MAX_TOTAL_MS / 1000.0) * self.sample_rate).ceil() as usize + 4;
        self.buf_l = vec![0.0; cap.max(4)];
        self.buf_r = vec![0.0; cap.max(4)];
        self.pos = 0;
    }
}

impl Effect for WowFlutter {
    fn init(&mut self, sample_rate: u32, _internal_buffer_size: usize) {
        self.resize(sample_rate);
    }

    fn on_change_sample_rate(&mut self, sample_rate: u32) {
        self.resize(sample_rate);
    }

    fn on_start_processing(&mut self) {
        self.base_ms
            .read_command(&mut self.command_readers.set_base_ms);
        self.depth_ms
            .read_command(&mut self.command_readers.set_depth_ms);
        self.wow_rate_hz
            .read_command(&mut self.command_readers.set_wow_rate_hz);
        self.flutter_rate_hz
            .read_command(&mut self.command_readers.set_flutter_rate_hz);
        self.mix.read_command(&mut self.command_readers.set_mix);
    }

    fn process(&mut self, input: &mut [Frame], dt: f64, info: &Info) {
        self.base_ms.update(dt * input.len() as f64, info);
        self.depth_ms.update(dt * input.len() as f64, info);
        self.wow_rate_hz.update(dt * input.len() as f64, info);
        self.flutter_rate_hz.update(dt * input.len() as f64, info);
        self.mix.update(dt * input.len() as f64, info);
        let num_frames = input.len();
        let dt_sample = 1.0 / self.sample_rate;
        let cap = self.buf_l.len();
        let max_total_samples = (cap as f64 - 2.0).max(0.0);
        for (i, frame) in input.iter_mut().enumerate() {
            let t = (i + 1) as f64 / num_frames as f64;
            let base_ms = self.base_ms.interpolated_value(t).clamp(0.0, MAX_TOTAL_MS);
            let depth_ms = self.depth_ms.interpolated_value(t).clamp(0.0, MAX_TOTAL_MS);
            let wow_rate = self.wow_rate_hz.interpolated_value(t).max(0.0);
            let flutter_rate = self.flutter_rate_hz.interpolated_value(t).max(0.0);
            let mix = self.mix.interpolated_value(t).0.clamp(0.0, 1.0);

            self.wow_phase = (self.wow_phase + std::f64::consts::TAU * wow_rate * dt_sample)
                % std::f64::consts::TAU;
            self.flutter_phase = (self.flutter_phase
                + std::f64::consts::TAU * flutter_rate * dt_sample)
                % std::f64::consts::TAU;
            // In [-1, 1]: 0.6*sin(wow) + 0.4*sin(flutter), bounded exactly by
            // construction (weights sum to 1.0). Centered at 0 so `depth = 0`
            // leaves `lag` at exactly `base_samples`, regardless of phase.
            let m = 0.6 * self.wow_phase.sin() + 0.4 * self.flutter_phase.sin();
            let base_samples = (base_ms / 1000.0) * self.sample_rate;
            let depth_samples = (depth_ms / 1000.0) * self.sample_rate;
            let lag = (base_samples + depth_samples * m).clamp(0.0, max_total_samples);

            // Push the newest sample into both channel lines, then read the
            // shared modulated lag back on each.
            self.buf_l[self.pos] = frame.left;
            self.buf_r[self.pos] = frame.right;
            self.pos = (self.pos + 1) % cap;
            let wet = Frame::new(
                fractional_read(&self.buf_l, self.pos, lag),
                fractional_read(&self.buf_r, self.pos, lag),
            );
            *frame = wet * mix.sqrt() + *frame * (1.0 - mix).sqrt();
        }
    }
}

/// Controls a [`WowFlutter`] effect.
#[derive(Debug)]
pub struct WowFlutterHandle {
    command_writers: CommandWriters,
}

impl WowFlutterHandle {
    /// Sets the fixed base delay, in milliseconds.
    pub fn set_base_ms(&mut self, base_ms: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_base_ms.write(ValueChangeCommand {
            target: base_ms.into(),
            tween,
        });
    }

    /// Sets the peak modulation deviation from the base delay, in milliseconds.
    pub fn set_depth_ms(&mut self, depth_ms: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_depth_ms.write(ValueChangeCommand {
            target: depth_ms.into(),
            tween,
        });
    }

    /// Sets the slow "wow" drift rate, in Hz.
    pub fn set_wow_rate_hz(&mut self, wow_rate_hz: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers
            .set_wow_rate_hz
            .write(ValueChangeCommand {
                target: wow_rate_hz.into(),
                tween,
            });
    }

    /// Sets the fast "flutter" rate, in Hz.
    pub fn set_flutter_rate_hz(&mut self, flutter_rate_hz: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers
            .set_flutter_rate_hz
            .write(ValueChangeCommand {
                target: flutter_rate_hz.into(),
                tween,
            });
    }

    /// Sets the dry/wet blend.
    pub fn set_mix(&mut self, mix: impl Into<Value<Mix>>, tween: Tween) {
        self.command_writers.set_mix.write(ValueChangeCommand {
            target: mix.into(),
            tween,
        });
    }
}

kira::command_writers_and_readers! {
    set_base_ms: ValueChangeCommand<f64>,
    set_depth_ms: ValueChangeCommand<f64>,
    set_wow_rate_hz: ValueChangeCommand<f64>,
    set_flutter_rate_hz: ValueChangeCommand<f64>,
    set_mix: ValueChangeCommand<Mix>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::info::MockInfoBuilder;

    fn run(builder: WowFlutterBuilder, input: &[f32]) -> Vec<Frame> {
        let (mut effect, _handle) = builder.build();
        effect.init(44_100, input.len());
        let mut frames: Vec<Frame> = input.iter().map(|&x| Frame::from_mono(x)).collect();
        let info = MockInfoBuilder::new().build();
        effect.process(&mut frames, 1.0 / 44_100.0, &info);
        frames
    }

    #[test]
    fn neutral_at_zero_mix_is_bit_exact() {
        let input: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.6).collect();
        let out = run(
            WowFlutterBuilder::new()
                .base_ms(12.0)
                .depth_ms(4.0)
                .wow_rate_hz(1.5)
                .flutter_rate_hz(9.0),
            &input,
        );
        for (a, b) in input.iter().zip(out.iter()) {
            assert_eq!(*a, b.left);
            assert_eq!(*a, b.right);
        }
    }

    #[test]
    fn depth_zero_is_a_pure_delay_of_base_ms() {
        // A unit impulse delayed by exactly `base_ms` should land at sample
        // `round(base_ms/1000 * sample_rate)` with depth=0 collapsing the
        // modulation to nothing, regardless of LFO phase.
        let sample_rate = 44_100.0;
        let base_ms = 10.0;
        let mut input = vec![0.0f32; 4410];
        input[0] = 1.0;
        let out = run(
            WowFlutterBuilder::new()
                .base_ms(base_ms)
                .depth_ms(0.0)
                .wow_rate_hz(1.7)
                .flutter_rate_hz(11.0)
                .mix(Mix::WET),
            &input,
        );
        let expected = (base_ms / 1000.0 * sample_rate).round() as usize;
        let (peak_i, _) = out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.left.abs().total_cmp(&b.left.abs()))
            .unwrap();
        assert!(
            (peak_i as i64 - expected as i64).abs() <= 1,
            "depth=0 delay peak at {peak_i}, expected near {expected}"
        );
    }

    #[test]
    fn depth_and_mix_wobble_a_steady_tone() {
        let input: Vec<f32> = (0..4410).map(|i| (i as f32 * 0.08).sin() * 0.7).collect();
        let out = run(
            WowFlutterBuilder::new()
                .base_ms(10.0)
                .depth_ms(4.0)
                .wow_rate_hz(1.0)
                .flutter_rate_hz(6.0)
                .mix(Mix::WET),
            &input,
        );
        assert_ne!(out.iter().map(|f| f.left).collect::<Vec<_>>(), input);
        for f in &out {
            assert!(f.left.is_finite() && f.right.is_finite());
            assert!(f.left.abs() <= 1.5, "unexpectedly loud: {}", f.left);
        }
    }

    #[test]
    fn deterministic_given_the_same_seed() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.04).sin() * 0.5).collect();
        let a = run(
            WowFlutterBuilder::new()
                .base_ms(10.0)
                .depth_ms(5.0)
                .mix(0.8)
                .seed(3),
            &input,
        );
        let b = run(
            WowFlutterBuilder::new()
                .base_ms(10.0)
                .depth_ms(5.0)
                .mix(0.8)
                .seed(3),
            &input,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn no_allocation_shaped_process_handles_varied_block_sizes() {
        for block in [1usize, 3, 64, 999] {
            let input = vec![0.2f32; block];
            let out = run(
                WowFlutterBuilder::new()
                    .base_ms(10.0)
                    .depth_ms(4.0)
                    .mix(0.5),
                &input,
            );
            assert_eq!(out.len(), block);
        }
    }
}
