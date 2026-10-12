//! LFO amplitude modulation: a smooth sine tremolo or a hard on/off gate.
//! Kira has no tremolo/gate effect (see the module doc on [`super`]).

use kira::command::ValueChangeCommand;
use kira::effect::{Effect, EffectBuilder};
use kira::info::Info;
use kira::{Frame, Parameter, Tween, Value};

use crate::synth::{noise, Lcg};

/// The modulation shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TremoloShape {
    /// Smooth `0.5·(1 − cos(phase))` — classic tremolo.
    #[default]
    Sine,
    /// Hard on/off — a rhythmic gate.
    Gate,
}

/// Configures a [`TremoloGate`] effect.
///
/// Neutral (identity) at `depth = 0.0` (the default): the per-sample gain is
/// `1.0 − depth·g_mod`, which is exactly `1.0` for any `g_mod` when
/// `depth == 0.0`, for either shape and at any phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TremoloGateBuilder {
    /// Modulation rate in Hz.
    pub rate_hz: Value<f64>,
    /// `0.0` (the default) = no effect; `1.0` = full tremolo swing / a fully
    /// closing gate.
    pub depth: Value<f64>,
    /// Sine tremolo or a hard gate.
    pub shape: TremoloShape,
    /// Deterministic seed for the LFO's initial phase.
    pub seed: u32,
}

impl TremoloGateBuilder {
    /// A builder with `depth` at its neutral (identity) default of `0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the modulation rate, in Hz (see the `rate_hz` field doc).
    #[must_use = "This method consumes self and returns a modified TremoloGateBuilder, so the return value should be used"]
    pub fn rate_hz(self, rate_hz: impl Into<Value<f64>>) -> Self {
        Self {
            rate_hz: rate_hz.into(),
            ..self
        }
    }

    /// Sets the modulation depth (see the `depth` field doc).
    #[must_use = "This method consumes self and returns a modified TremoloGateBuilder, so the return value should be used"]
    pub fn depth(self, depth: impl Into<Value<f64>>) -> Self {
        Self {
            depth: depth.into(),
            ..self
        }
    }

    /// Sets the modulation shape (sine tremolo or a hard gate).
    #[must_use = "This method consumes self and returns a modified TremoloGateBuilder, so the return value should be used"]
    pub fn shape(self, shape: TremoloShape) -> Self {
        Self { shape, ..self }
    }

    /// Sets the deterministic seed for the LFO's initial phase.
    #[must_use = "This method consumes self and returns a modified TremoloGateBuilder, so the return value should be used"]
    pub fn seed(self, seed: u32) -> Self {
        Self { seed, ..self }
    }
}

impl Default for TremoloGateBuilder {
    fn default() -> Self {
        Self {
            rate_hz: Value::Fixed(5.0),
            depth: Value::Fixed(0.0),
            shape: TremoloShape::Sine,
            seed: 0,
        }
    }
}

impl EffectBuilder for TremoloGateBuilder {
    type Handle = TremoloGateHandle;

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        let (command_writers, command_readers) = command_writers_and_readers();
        let mut lcg = Lcg(self.seed ^ 0x7E60_C0DE);
        let phase = (f64::from(noise(&mut lcg)) * 0.5 + 0.5) * std::f64::consts::TAU;
        (
            Box::new(TremoloGate {
                command_readers,
                rate_hz: Parameter::new(self.rate_hz, 5.0),
                depth: Parameter::new(self.depth, 0.0),
                shape: self.shape,
                phase,
                sample_rate: 44_100.0,
            }),
            TremoloGateHandle { command_writers },
        )
    }
}

struct TremoloGate {
    command_readers: CommandReaders,
    rate_hz: Parameter<f64>,
    depth: Parameter<f64>,
    shape: TremoloShape,
    phase: f64,
    sample_rate: f64,
}

impl Effect for TremoloGate {
    fn init(&mut self, sample_rate: u32, _internal_buffer_size: usize) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
    }

    fn on_change_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
    }

    fn on_start_processing(&mut self) {
        self.rate_hz
            .read_command(&mut self.command_readers.set_rate_hz);
        self.depth.read_command(&mut self.command_readers.set_depth);
        if let Some(shape) = self.command_readers.set_shape.read() {
            self.shape = shape;
        }
    }

    fn process(&mut self, input: &mut [Frame], dt: f64, info: &Info) {
        self.rate_hz.update(dt * input.len() as f64, info);
        self.depth.update(dt * input.len() as f64, info);
        let num_frames = input.len();
        let dt_sample = 1.0 / self.sample_rate;
        for (i, frame) in input.iter_mut().enumerate() {
            let t = (i + 1) as f64 / num_frames as f64;
            let rate = self.rate_hz.interpolated_value(t).max(0.0);
            let depth = self.depth.interpolated_value(t).clamp(0.0, 1.0) as f32;

            self.phase =
                (self.phase + std::f64::consts::TAU * rate * dt_sample) % std::f64::consts::TAU;
            let g_mod = match self.shape {
                TremoloShape::Sine => (0.5 * (1.0 - self.phase.cos())) as f32,
                TremoloShape::Gate => {
                    if self.phase.sin() >= 0.0 {
                        0.0
                    } else {
                        1.0
                    }
                }
            };
            let gain = 1.0 - depth * g_mod;
            *frame *= gain;
        }
    }
}

/// Controls a [`TremoloGate`] effect.
#[derive(Debug)]
pub struct TremoloGateHandle {
    command_writers: CommandWriters,
}

impl TremoloGateHandle {
    /// Sets the modulation rate, in Hz.
    pub fn set_rate_hz(&mut self, rate_hz: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_rate_hz.write(ValueChangeCommand {
            target: rate_hz.into(),
            tween,
        });
    }

    /// Sets the modulation depth.
    pub fn set_depth(&mut self, depth: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_depth.write(ValueChangeCommand {
            target: depth.into(),
            tween,
        });
    }

    /// Sets the modulation shape (sine tremolo or a hard gate).
    pub fn set_shape(&mut self, shape: TremoloShape) {
        self.command_writers.set_shape.write(shape);
    }
}

kira::command_writers_and_readers! {
    set_rate_hz: ValueChangeCommand<f64>,
    set_depth: ValueChangeCommand<f64>,
    set_shape: TremoloShape,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::info::MockInfoBuilder;

    fn run(builder: TremoloGateBuilder, input: &[f32]) -> Vec<Frame> {
        let (mut effect, _handle) = builder.build();
        effect.init(44_100, input.len());
        let mut frames: Vec<Frame> = input.iter().map(|&x| Frame::from_mono(x)).collect();
        let info = MockInfoBuilder::new().build();
        effect.process(&mut frames, 1.0 / 44_100.0, &info);
        frames
    }

    #[test]
    fn neutral_at_zero_depth_is_bit_exact_for_both_shapes() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.02).sin() * 0.6).collect();
        for shape in [TremoloShape::Sine, TremoloShape::Gate] {
            let out = run(TremoloGateBuilder::new().rate_hz(6.0).shape(shape), &input);
            for (a, b) in input.iter().zip(out.iter()) {
                assert_eq!(*a, b.left, "shape {shape:?} not neutral at depth 0");
            }
        }
    }

    #[test]
    fn full_depth_gate_produces_true_silence_half_the_cycle() {
        let input = vec![1.0f32; 44_100];
        let out = run(
            TremoloGateBuilder::new()
                .rate_hz(1.0)
                .depth(1.0)
                .shape(TremoloShape::Gate),
            &input,
        );
        let silent = out.iter().filter(|f| f.left.abs() < 1e-6).count();
        assert!(
            silent > 15_000,
            "gate did not close for roughly half the cycle: {silent}/44100"
        );
    }

    #[test]
    fn sine_tremolo_stays_bounded_and_deterministic() {
        let input: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.05).sin() * 0.8).collect();
        let a = run(
            TremoloGateBuilder::new().rate_hz(4.0).depth(0.7).seed(9),
            &input,
        );
        let b = run(
            TremoloGateBuilder::new().rate_hz(4.0).depth(0.7).seed(9),
            &input,
        );
        assert_eq!(a, b);
        for f in &a {
            assert!(f.left.abs() <= 0.8 + 1e-4);
        }
    }
}
