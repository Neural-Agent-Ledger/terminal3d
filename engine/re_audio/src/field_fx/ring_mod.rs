//! Ring modulation: multiply the signal by an internal sine carrier. Kira has
//! no audio-rate modulation effect (see the module doc on [`super`]).

use kira::command::ValueChangeCommand;
use kira::effect::{Effect, EffectBuilder};
use kira::info::Info;
use kira::{Frame, Mix, Parameter, Tween, Value};

use crate::synth::{noise, Lcg};

/// Configures a [`RingMod`] effect.
///
/// Neutral (identity) at `mix = Mix::DRY` (the default) — `freq_hz` is
/// irrelevant while `mix` is 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingModBuilder {
    /// Carrier frequency in Hz.
    pub freq_hz: Value<f64>,
    /// Dry/wet blend. `0.0` (the default) is bit-exact passthrough.
    pub mix: Value<Mix>,
    /// Deterministic seed for the carrier's initial phase, so several
    /// simultaneous ring-mod instances (e.g. one per bus) don't lock into an
    /// identical, phase-coherent multiply. Two builders with the same seed
    /// always start at the same phase.
    pub seed: u32,
}

impl RingModBuilder {
    /// A builder with all fields at their neutral (bit-exact passthrough) defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the carrier frequency, in Hz (see the `freq_hz` field doc).
    #[must_use = "This method consumes self and returns a modified RingModBuilder, so the return value should be used"]
    pub fn freq_hz(self, freq_hz: impl Into<Value<f64>>) -> Self {
        Self {
            freq_hz: freq_hz.into(),
            ..self
        }
    }

    /// Sets the dry/wet blend (see the `mix` field doc).
    #[must_use = "This method consumes self and returns a modified RingModBuilder, so the return value should be used"]
    pub fn mix(self, mix: impl Into<Value<Mix>>) -> Self {
        Self {
            mix: mix.into(),
            ..self
        }
    }

    /// Sets the deterministic seed for the carrier's initial phase (see the
    /// `seed` field doc).
    #[must_use = "This method consumes self and returns a modified RingModBuilder, so the return value should be used"]
    pub fn seed(self, seed: u32) -> Self {
        Self { seed, ..self }
    }
}

impl Default for RingModBuilder {
    fn default() -> Self {
        Self {
            freq_hz: Value::Fixed(200.0),
            mix: Value::Fixed(Mix::DRY),
            seed: 0,
        }
    }
}

impl EffectBuilder for RingModBuilder {
    type Handle = RingModHandle;

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        let (command_writers, command_readers) = command_writers_and_readers();
        let mut lcg = Lcg(self.seed ^ 0xA1CE_F00D);
        let phase = (f64::from(noise(&mut lcg)) * 0.5 + 0.5) * std::f64::consts::TAU;
        (
            Box::new(RingMod {
                command_readers,
                freq_hz: Parameter::new(self.freq_hz, 200.0),
                mix: Parameter::new(self.mix, Mix::DRY),
                phase,
                sample_rate: 44_100.0,
            }),
            RingModHandle { command_writers },
        )
    }
}

struct RingMod {
    command_readers: CommandReaders,
    freq_hz: Parameter<f64>,
    mix: Parameter<Mix>,
    /// Carrier phase in radians, wrapped into `[0, TAU)`.
    phase: f64,
    sample_rate: f64,
}

impl Effect for RingMod {
    fn init(&mut self, sample_rate: u32, _internal_buffer_size: usize) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
    }

    fn on_change_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
    }

    fn on_start_processing(&mut self) {
        self.freq_hz
            .read_command(&mut self.command_readers.set_freq_hz);
        self.mix.read_command(&mut self.command_readers.set_mix);
    }

    fn process(&mut self, input: &mut [Frame], dt: f64, info: &Info) {
        self.freq_hz.update(dt * input.len() as f64, info);
        self.mix.update(dt * input.len() as f64, info);
        let num_frames = input.len();
        let dt_sample = 1.0 / self.sample_rate;
        for (i, frame) in input.iter_mut().enumerate() {
            let t = (i + 1) as f64 / num_frames as f64;
            let freq = self.freq_hz.interpolated_value(t).max(0.0);
            let mix = self.mix.interpolated_value(t).0.clamp(0.0, 1.0);

            self.phase =
                (self.phase + std::f64::consts::TAU * freq * dt_sample) % std::f64::consts::TAU;
            let carrier = self.phase.sin() as f32;
            let wet = *frame * carrier;
            *frame = wet * mix.sqrt() + *frame * (1.0 - mix).sqrt();
        }
    }
}

/// Controls a [`RingMod`] effect.
#[derive(Debug)]
pub struct RingModHandle {
    command_writers: CommandWriters,
}

impl RingModHandle {
    /// Sets the carrier frequency (Hz).
    pub fn set_freq_hz(&mut self, freq_hz: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_freq_hz.write(ValueChangeCommand {
            target: freq_hz.into(),
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
    set_freq_hz: ValueChangeCommand<f64>,
    set_mix: ValueChangeCommand<Mix>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::info::MockInfoBuilder;

    fn run(builder: RingModBuilder, input: &[f32]) -> Vec<Frame> {
        let (mut effect, _handle) = builder.build();
        effect.init(44_100, input.len());
        let mut frames: Vec<Frame> = input.iter().map(|&x| Frame::from_mono(x)).collect();
        let info = MockInfoBuilder::new().build();
        effect.process(&mut frames, 1.0 / 44_100.0, &info);
        frames
    }

    #[test]
    fn neutral_at_zero_mix_is_bit_exact() {
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
        let out = run(RingModBuilder::new().freq_hz(880.0), &input);
        for (a, b) in input.iter().zip(out.iter()) {
            assert_eq!(*a, b.left);
        }
    }

    #[test]
    fn full_wet_produces_a_sum_and_difference_spectrum_not_the_input() {
        // A pure tone ring-modulated by another pure tone is NOT the input
        // signal (it becomes sum/difference frequencies) — assert it differs
        // substantially, a cheap proxy for "modulation actually happened".
        let input: Vec<f32> = (0..4410).map(|i| (i as f32 * 0.05).sin() * 0.7).collect();
        let out = run(RingModBuilder::new().freq_hz(300.0).mix(Mix::WET), &input);
        let diff: f32 = input
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b.left).abs())
            .sum();
        assert!(diff > 100.0, "ring mod barely changed the signal: {diff}");
        for f in &out {
            assert!(f.left.is_finite());
        }
    }

    #[test]
    fn deterministic_given_the_same_seed_and_params() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.03).cos() * 0.4).collect();
        let a = run(
            RingModBuilder::new().freq_hz(150.0).mix(0.6).seed(7),
            &input,
        );
        let b = run(
            RingModBuilder::new().freq_hz(150.0).mix(0.6).seed(7),
            &input,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_start_at_different_phases() {
        let input = vec![1.0f32; 8];
        let a = run(
            RingModBuilder::new().freq_hz(0.0).mix(Mix::WET).seed(1),
            &input,
        );
        let b = run(
            RingModBuilder::new().freq_hz(0.0).mix(Mix::WET).seed(99),
            &input,
        );
        assert_ne!(a, b, "different seeds produced identical carrier phase");
    }
}
