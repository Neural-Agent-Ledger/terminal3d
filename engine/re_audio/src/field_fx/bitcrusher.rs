//! `Crush` — bit-depth quantization + sample & hold decimation + seeded
//! dropout gaps ("lo-fi crusher"). No kira built-in does any of this: its
//! `Distortion` is waveshaping only (hard/soft clip), never quantization,
//! decimation or gating (see the module doc on [`super`]).
//!
//! Feeds the field chain's `CR` slot: `bits` ranges 16 → 6, `hold` in samples
//! ranges 1 → 8, `dropout` probability ranges 0 → 0.25 with 10-60 ms gaps,
//! plus a dry/wet `mix`.

use kira::command::ValueChangeCommand;
use kira::effect::{Effect, EffectBuilder};
use kira::info::Info;
use kira::{Frame, Mix, Parameter, Tween, Value};

use crate::synth::{noise, Lcg};

/// Configures a [`Bitcrusher`] ("Crush") effect.
///
/// Neutral (identity) two ways: `mix = Mix::DRY` (the default) is the ordinary
/// dry/wet bypass, and — regardless of `mix` — `bits >= 16 && hold <= 1 &&
/// dropout <= 0` (also all defaults) is a dedicated fast-path that returns
/// the input untouched, bit-exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitcrusherBuilder {
    /// Effective bit depth in `[1, 16]`. Lower = coarser quantization steps.
    pub bits: Value<f64>,
    /// Sample & hold window in samples (`1.0` = no reduction, `8.0` = every
    /// eighth sample captured and held). Values `< 1` are clamped to `1`.
    pub hold: Value<f64>,
    /// Per-capture probability `[0, 1]` of opening a silent dropout gap
    /// (10–60 ms, seeded).
    pub dropout: Value<f64>,
    /// Dry/wet blend. `0.0` (the default) is bit-exact passthrough.
    pub mix: Value<Mix>,
    /// Deterministic seed for the dropout-gap RNG.
    pub seed: u32,
}

impl BitcrusherBuilder {
    /// A builder with all fields at their neutral (bit-exact passthrough) defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the effective bit depth (see the `bits` field doc).
    #[must_use = "This method consumes self and returns a modified BitcrusherBuilder, so the return value should be used"]
    pub fn bits(self, bits: impl Into<Value<f64>>) -> Self {
        Self {
            bits: bits.into(),
            ..self
        }
    }

    /// Sets the sample & hold window (see the `hold` field doc).
    #[must_use = "This method consumes self and returns a modified BitcrusherBuilder, so the return value should be used"]
    pub fn hold(self, hold: impl Into<Value<f64>>) -> Self {
        Self {
            hold: hold.into(),
            ..self
        }
    }

    /// Sets the per-capture dropout-gap probability (see the `dropout` field doc).
    #[must_use = "This method consumes self and returns a modified BitcrusherBuilder, so the return value should be used"]
    pub fn dropout(self, dropout: impl Into<Value<f64>>) -> Self {
        Self {
            dropout: dropout.into(),
            ..self
        }
    }

    /// Sets the dry/wet blend (see the `mix` field doc).
    #[must_use = "This method consumes self and returns a modified BitcrusherBuilder, so the return value should be used"]
    pub fn mix(self, mix: impl Into<Value<Mix>>) -> Self {
        Self {
            mix: mix.into(),
            ..self
        }
    }

    /// Sets the deterministic seed for the dropout-gap RNG.
    #[must_use = "This method consumes self and returns a modified BitcrusherBuilder, so the return value should be used"]
    pub fn seed(self, seed: u32) -> Self {
        Self { seed, ..self }
    }
}

impl Default for BitcrusherBuilder {
    fn default() -> Self {
        Self {
            bits: Value::Fixed(16.0),
            hold: Value::Fixed(1.0),
            dropout: Value::Fixed(0.0),
            mix: Value::Fixed(Mix::DRY),
            seed: 0,
        }
    }
}

impl EffectBuilder for BitcrusherBuilder {
    type Handle = BitcrusherHandle;

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        let (command_writers, command_readers) = command_writers_and_readers();
        (
            Box::new(Bitcrusher {
                command_readers,
                bits: Parameter::new(self.bits, 16.0),
                hold: Parameter::new(self.hold, 1.0),
                dropout: Parameter::new(self.dropout, 0.0),
                mix: Parameter::new(self.mix, Mix::DRY),
                phase: 0.0,
                held: Frame::ZERO,
                gap_remaining: 0,
                lcg: Lcg(self.seed ^ 0xC2B2_AE35),
                sample_rate: 44_100.0,
            }),
            BitcrusherHandle { command_writers },
        )
    }
}

struct Bitcrusher {
    command_readers: CommandReaders,
    bits: Parameter<f64>,
    hold: Parameter<f64>,
    dropout: Parameter<f64>,
    mix: Parameter<Mix>,
    /// Sample & hold phase accumulator in `[0, 1)`.
    phase: f64,
    /// The held (quantized, or silent during a dropout gap) sample.
    held: Frame,
    /// Samples remaining in an open dropout gap.
    gap_remaining: u32,
    lcg: Lcg,
    sample_rate: f64,
}

#[inline]
fn quantize(x: f32, bits: f64) -> f32 {
    let bits = bits.clamp(1.0, 16.0);
    let levels = 2f64.powf(bits) - 1.0;
    (((f64::from(x) * 0.5 + 0.5).clamp(0.0, 1.0) * levels).round() / levels * 2.0 - 1.0) as f32
}

impl Bitcrusher {
    #[inline]
    fn roll(&mut self) -> f32 {
        noise(&mut self.lcg) * 0.5 + 0.5
    }
}

impl Effect for Bitcrusher {
    fn init(&mut self, sample_rate: u32, _internal_buffer_size: usize) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
    }

    fn on_change_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
    }

    fn on_start_processing(&mut self) {
        self.bits.read_command(&mut self.command_readers.set_bits);
        self.hold.read_command(&mut self.command_readers.set_hold);
        self.dropout
            .read_command(&mut self.command_readers.set_dropout);
        self.mix.read_command(&mut self.command_readers.set_mix);
    }

    fn process(&mut self, input: &mut [Frame], dt: f64, info: &Info) {
        self.bits.update(dt * input.len() as f64, info);
        self.hold.update(dt * input.len() as f64, info);
        self.dropout.update(dt * input.len() as f64, info);
        self.mix.update(dt * input.len() as f64, info);
        let num_frames = input.len();
        for (i, frame) in input.iter_mut().enumerate() {
            let t = (i + 1) as f64 / num_frames as f64;
            let bits = self.bits.interpolated_value(t);
            let hold = self.hold.interpolated_value(t).max(1.0);
            let dropout = self.dropout.interpolated_value(t).clamp(0.0, 1.0);
            let mix = self.mix.interpolated_value(t).0.clamp(0.0, 1.0);

            // Dedicated bit-exact fast path (the documented "no-op" corner).
            if bits >= 16.0 && hold <= 1.0 && dropout <= 0.0 {
                self.phase = 0.0;
                self.gap_remaining = 0;
                continue;
            }

            if self.gap_remaining > 0 {
                self.gap_remaining -= 1;
                self.held = Frame::ZERO;
            } else {
                self.phase += 1.0 / hold;
                if self.phase >= 1.0 {
                    self.phase -= self.phase.floor();
                    if dropout > 0.0 && f64::from(self.roll()) < dropout {
                        let gap_ms = 10.0 + f64::from(self.roll()) * 50.0; // [10, 60)
                        self.gap_remaining = ((gap_ms / 1000.0) * self.sample_rate) as u32;
                        self.held = Frame::ZERO;
                    } else {
                        self.held =
                            Frame::new(quantize(frame.left, bits), quantize(frame.right, bits));
                    }
                }
            }
            *frame = self.held * mix.sqrt() + *frame * (1.0 - mix).sqrt();
        }
    }
}

/// Controls a [`Bitcrusher`] ("Crush") effect.
#[derive(Debug)]
pub struct BitcrusherHandle {
    command_writers: CommandWriters,
}

impl BitcrusherHandle {
    /// Sets the effective bit depth (`[1, 16]`).
    pub fn set_bits(&mut self, bits: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_bits.write(ValueChangeCommand {
            target: bits.into(),
            tween,
        });
    }

    /// Sets the sample & hold window (samples).
    pub fn set_hold(&mut self, hold: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_hold.write(ValueChangeCommand {
            target: hold.into(),
            tween,
        });
    }

    /// Sets the per-capture dropout-gap probability.
    pub fn set_dropout(&mut self, dropout: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_dropout.write(ValueChangeCommand {
            target: dropout.into(),
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
    set_bits: ValueChangeCommand<f64>,
    set_hold: ValueChangeCommand<f64>,
    set_dropout: ValueChangeCommand<f64>,
    set_mix: ValueChangeCommand<Mix>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::info::MockInfoBuilder;

    fn run(builder: BitcrusherBuilder, input: &[f32]) -> Vec<Frame> {
        let (mut effect, _handle) = builder.build();
        effect.init(44_100, input.len());
        let mut frames: Vec<Frame> = input.iter().map(|&x| Frame::from_mono(x)).collect();
        let info = MockInfoBuilder::new().build();
        effect.process(&mut frames, 1.0 / 44_100.0, &info);
        frames
    }

    #[test]
    fn neutral_at_zero_mix_is_bit_exact() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.13).sin() * 0.8).collect();
        let out = run(BitcrusherBuilder::new().bits(3.0).hold(20.0), &input);
        for (a, b) in input.iter().zip(out.iter()) {
            assert_eq!(*a, b.left);
            assert_eq!(*a, b.right);
        }
    }

    #[test]
    fn documented_noop_corner_is_bit_exact_even_at_full_wet() {
        // bits=16, hold=1, dropout=0 must be an identity regardless of mix —
        // the dedicated fast path, not merely "low quantization error".
        let input: Vec<f32> = (0..500).map(|i| (i as f32 * 0.021).sin() * 0.73).collect();
        let out = run(BitcrusherBuilder::new().mix(Mix::WET), &input);
        assert_eq!(
            input
                .iter()
                .map(|&x| Frame::from_mono(x))
                .collect::<Vec<_>>(),
            out
        );
    }

    #[test]
    fn full_wet_crushes_toward_a_small_step_count() {
        let input: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.05).sin() * 0.9).collect();
        let out = run(
            BitcrusherBuilder::new().bits(2.0).hold(1.0).mix(Mix::WET),
            &input,
        );
        let distinct: std::collections::BTreeSet<i64> = out
            .iter()
            .map(|f| (f.left * 1_000_000.0).round() as i64)
            .collect();
        assert!(
            distinct.len() <= 8,
            "expected a small step count, got {}",
            distinct.len()
        );
    }

    #[test]
    fn dropout_opens_silent_gaps() {
        let input = vec![1.0f32; 44_100];
        let out = run(
            BitcrusherBuilder::new()
                .bits(8.0)
                .hold(2.0)
                .dropout(0.9)
                .mix(Mix::WET)
                .seed(3),
            &input,
        );
        let silent = out.iter().filter(|f| f.left == 0.0).count();
        assert!(
            silent > 1000,
            "a 0.9 dropout probability produced almost no gaps: {silent}/44100"
        );
    }

    #[test]
    fn no_allocation_shaped_process_never_panics_across_block_sizes() {
        for block in [1usize, 7, 64, 512] {
            let input = vec![0.3f32; block];
            let out = run(
                BitcrusherBuilder::new()
                    .bits(5.0)
                    .hold(3.0)
                    .dropout(0.1)
                    .mix(0.5),
                &input,
            );
            assert_eq!(out.len(), block);
            for f in &out {
                assert!(f.left.is_finite() && f.right.is_finite());
            }
        }
    }

    #[test]
    fn deterministic_given_the_same_input_and_params() {
        let input: Vec<f32> = (0..500).map(|i| (i as f32 * 0.07).sin() * 0.6).collect();
        let a = run(
            BitcrusherBuilder::new()
                .bits(4.0)
                .hold(6.0)
                .dropout(0.2)
                .mix(0.7)
                .seed(9),
            &input,
        );
        let b = run(
            BitcrusherBuilder::new()
                .bits(4.0)
                .hold(6.0)
                .dropout(0.2)
                .mix(0.7)
                .seed(9),
            &input,
        );
        assert_eq!(a, b);
    }
}
