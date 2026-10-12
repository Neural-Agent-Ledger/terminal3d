//! Stereo collapse / widen via mid/side reconstruction. Kira's reverb has a
//! `stereo_width` knob but only for its own wet tail; there is no
//! general-purpose stereo-image effect on the dry signal (see the module doc
//! on [`super`]).

use kira::command::ValueChangeCommand;
use kira::effect::{Effect, EffectBuilder};
use kira::info::Info;
use kira::{Frame, Parameter, Tween, Value};

/// Configures a [`StereoWidth`] effect.
///
/// Neutral (identity, modulo float rounding of the mid/side round-trip) at
/// `width = 1.0` (the default): `mid = (l+r)/2`, `side = (l-r)/2`, and
/// `l' = mid+side`, `r' = mid-side` reproduce `l`/`r` exactly in real
/// arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoWidthBuilder {
    /// `0.0` = fully collapsed to mono, `1.0` = untouched (the default),
    /// `2.0` = doubled side-channel width.
    pub width: Value<f64>,
}

impl StereoWidthBuilder {
    /// A builder with `width` at its neutral (identity) default of `1.0`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the stereo width (see the `width` field doc).
    #[must_use = "This method consumes self and returns a modified StereoWidthBuilder, so the return value should be used"]
    pub fn width(self, width: impl Into<Value<f64>>) -> Self {
        Self {
            width: width.into(),
        }
    }
}

impl Default for StereoWidthBuilder {
    fn default() -> Self {
        Self {
            width: Value::Fixed(1.0),
        }
    }
}

impl EffectBuilder for StereoWidthBuilder {
    type Handle = StereoWidthHandle;

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        let (command_writers, command_readers) = command_writers_and_readers();
        (
            Box::new(StereoWidth {
                command_readers,
                width: Parameter::new(self.width, 1.0),
            }),
            StereoWidthHandle { command_writers },
        )
    }
}

struct StereoWidth {
    command_readers: CommandReaders,
    width: Parameter<f64>,
}

impl Effect for StereoWidth {
    fn on_start_processing(&mut self) {
        self.width.read_command(&mut self.command_readers.set_width);
    }

    fn process(&mut self, input: &mut [Frame], dt: f64, info: &Info) {
        self.width.update(dt * input.len() as f64, info);
        let num_frames = input.len();
        for (i, frame) in input.iter_mut().enumerate() {
            let t = (i + 1) as f64 / num_frames as f64;
            let width = self.width.interpolated_value(t).max(0.0) as f32;
            let mid = (frame.left + frame.right) * 0.5;
            let side = (frame.left - frame.right) * 0.5 * width;
            *frame = Frame::new(mid + side, mid - side);
        }
    }
}

/// Controls a [`StereoWidth`] effect.
#[derive(Debug)]
pub struct StereoWidthHandle {
    command_writers: CommandWriters,
}

impl StereoWidthHandle {
    /// Sets the stereo width (see the `width` field doc on [`StereoWidthBuilder`]).
    pub fn set_width(&mut self, width: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers.set_width.write(ValueChangeCommand {
            target: width.into(),
            tween,
        });
    }
}

kira::command_writers_and_readers! {
    set_width: ValueChangeCommand<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::info::MockInfoBuilder;

    fn run(builder: StereoWidthBuilder, l: &[f32], r: &[f32]) -> Vec<Frame> {
        let (mut effect, _handle) = builder.build();
        effect.init(44_100, l.len());
        let mut frames: Vec<Frame> = l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| Frame::new(a, b))
            .collect();
        let info = MockInfoBuilder::new().build();
        effect.process(&mut frames, 1.0 / 44_100.0, &info);
        frames
    }

    #[test]
    fn neutral_at_width_one_is_exact_within_float_rounding() {
        let l: Vec<f32> = (0..512).map(|i| (i as f32 * 0.03).sin() * 0.7).collect();
        let r: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.03 + 0.4).cos() * 0.5)
            .collect();
        let out = run(StereoWidthBuilder::new(), &l, &r);
        for i in 0..l.len() {
            assert!((out[i].left - l[i]).abs() < 1e-6);
            assert!((out[i].right - r[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_width_collapses_to_identical_channels() {
        let l = vec![1.0f32, -0.5, 0.25];
        let r = vec![-1.0f32, 0.5, 0.75];
        let out = run(StereoWidthBuilder::new().width(0.0), &l, &r);
        for f in &out {
            assert!((f.left - f.right).abs() < 1e-6, "not collapsed: {f:?}");
        }
    }

    #[test]
    fn double_width_widens_the_side_signal() {
        let l = vec![1.0f32; 4];
        let r = vec![-1.0f32; 4];
        let neutral = run(StereoWidthBuilder::new(), &l, &r);
        let wide = run(StereoWidthBuilder::new().width(2.0), &l, &r);
        assert!(wide[0].left > neutral[0].left);
    }
}
