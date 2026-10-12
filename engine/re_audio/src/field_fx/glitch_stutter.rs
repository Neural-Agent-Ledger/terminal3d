//! `Hold` — deterministic, seeded buffer-repeat (stutter) or a sustained
//! captured-window loop (freeze). Nothing like it exists in kira: its reverb
//! has no freeze (feedback is an unclamped `Parameter`; `1.0` with live
//! input runs away), and it has no buffer-repeat at all (see the module doc
//! on [`super`]).
//!
//! Feeds the field chain's `HD` slot: `window` (20-500 ms), `mode`
//! (off/stutter/freeze), `mix` (0-0.7 in the field system; this effect itself
//! accepts the full `[0, 1]`).
//!
//! A short capture ring buffer always mirrors the recent dry signal. At
//! `mode = Off` (the default) or `mix = 0.0` the effect never reads the
//! capture buffer for output — it is a single `if` (or the mix-0 sqrt blend,
//! shared with every other field effect) away from a guaranteed-identity
//! passthrough. In `Stutter` mode it periodically decides (via a seeded LCG,
//! on a schedule advanced purely by a per-sample counter — never the wall
//! clock or block boundaries) whether to repeat a short recent slice. In
//! `Freeze` mode it captures one window on entry and loops it until the mode
//! changes away from `Freeze`.

use kira::command::ValueChangeCommand;
use kira::effect::{Effect, EffectBuilder};
use kira::info::Info;
use kira::{Frame, Mix, Parameter, Tween, Value};

use crate::synth::{noise, Lcg};

/// Longest window the capture buffer can hold; a larger `window`/`repeat_ms`
/// request is clamped.
const MAX_WINDOW_MS: f64 = 500.0;

/// The `HD` slot's operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HoldMode {
    /// Always passthrough — the fast bypass.
    #[default]
    Off,
    /// Periodically repeat a short recent slice.
    Stutter,
    /// Capture one window on entry and loop it until the mode changes.
    Freeze,
}

/// Configures a [`Hold`] effect.
///
/// Neutral (identity) at `mode = HoldMode::Off` (the default, a dedicated
/// bypass) or at `mix = 0.0` (also default — the ordinary dry/wet bypass
/// every stage shares).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoldBuilder {
    /// Operating mode: passthrough, periodic stutter, or a frozen loop.
    pub mode: HoldMode,
    /// Capture window, milliseconds (`[1, 500]`) — the freeze loop length,
    /// or (scaled) the stutter repeat-slice length.
    pub window_ms: Value<f64>,
    /// Average time between stutter decisions, milliseconds (`Stutter` mode
    /// only).
    pub interval_ms: Value<f64>,
    /// Dry/wet blend.
    pub mix: Value<Mix>,
    /// Deterministic seed driving the stutter decision RNG.
    pub seed: u32,
}

impl HoldBuilder {
    /// A builder with all fields at their neutral (passthrough) defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the operating mode (see the `mode` field doc).
    #[must_use = "This method consumes self and returns a modified HoldBuilder, so the return value should be used"]
    pub fn mode(self, mode: HoldMode) -> Self {
        Self { mode, ..self }
    }

    /// Sets the capture window length, in milliseconds (see the `window_ms` field doc).
    #[must_use = "This method consumes self and returns a modified HoldBuilder, so the return value should be used"]
    pub fn window_ms(self, window_ms: impl Into<Value<f64>>) -> Self {
        Self {
            window_ms: window_ms.into(),
            ..self
        }
    }

    /// Sets the average time between stutter decisions, in milliseconds
    /// (see the `interval_ms` field doc).
    #[must_use = "This method consumes self and returns a modified HoldBuilder, so the return value should be used"]
    pub fn interval_ms(self, interval_ms: impl Into<Value<f64>>) -> Self {
        Self {
            interval_ms: interval_ms.into(),
            ..self
        }
    }

    /// Sets the dry/wet blend (see the `mix` field doc).
    #[must_use = "This method consumes self and returns a modified HoldBuilder, so the return value should be used"]
    pub fn mix(self, mix: impl Into<Value<Mix>>) -> Self {
        Self {
            mix: mix.into(),
            ..self
        }
    }

    /// Sets the deterministic seed driving the stutter decision RNG.
    #[must_use = "This method consumes self and returns a modified HoldBuilder, so the return value should be used"]
    pub fn seed(self, seed: u32) -> Self {
        Self { seed, ..self }
    }
}

impl Default for HoldBuilder {
    fn default() -> Self {
        Self {
            mode: HoldMode::Off,
            window_ms: Value::Fixed(60.0),
            interval_ms: Value::Fixed(700.0),
            mix: Value::Fixed(Mix::DRY),
            seed: 0,
        }
    }
}

impl EffectBuilder for HoldBuilder {
    type Handle = HoldHandle;

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        let (command_writers, command_readers) = command_writers_and_readers();
        (
            Box::new(Hold {
                command_readers,
                mode: self.mode,
                window_ms: Parameter::new(self.window_ms, 60.0),
                interval_ms: Parameter::new(self.interval_ms, 700.0),
                mix: Parameter::new(self.mix, Mix::DRY),
                lcg: Lcg(self.seed ^ 0x9EC1_7C4E),
                sample_rate: 44_100.0,
                cap_l: Vec::new(),
                cap_r: Vec::new(),
                write_pos: 0,
                since_decision: 0,
                next_decision_in: 1,
                state: State::Passing,
                last_mode: HoldMode::Off,
            }),
            HoldHandle { command_writers },
        )
    }
}

#[derive(Default)]
enum State {
    #[default]
    Passing,
    Repeating {
        read_pos: usize,
        len: usize,
        remaining: usize,
    },
    Frozen {
        start: usize,
        len: usize,
        offset: usize,
    },
}

struct Hold {
    command_readers: CommandReaders,
    mode: HoldMode,
    window_ms: Parameter<f64>,
    interval_ms: Parameter<f64>,
    mix: Parameter<Mix>,
    lcg: Lcg,
    sample_rate: f64,
    cap_l: Vec<f32>,
    cap_r: Vec<f32>,
    write_pos: usize,
    since_decision: u64,
    next_decision_in: u64,
    state: State,
    last_mode: HoldMode,
}

impl Hold {
    fn resize(&mut self, sample_rate: u32) {
        self.sample_rate = f64::from(sample_rate).max(1.0);
        let cap = ((MAX_WINDOW_MS / 1000.0) * self.sample_rate).ceil() as usize + 4;
        self.cap_l = vec![0.0; cap.max(4)];
        self.cap_r = vec![0.0; cap.max(4)];
        self.write_pos = 0;
        self.state = State::Passing;
        self.since_decision = 0;
        self.next_decision_in = 1;
    }

    #[inline]
    fn roll(&mut self) -> f32 {
        noise(&mut self.lcg) * 0.5 + 0.5
    }
}

impl Effect for Hold {
    fn init(&mut self, sample_rate: u32, _internal_buffer_size: usize) {
        self.resize(sample_rate);
    }

    fn on_change_sample_rate(&mut self, sample_rate: u32) {
        self.resize(sample_rate);
    }

    fn on_start_processing(&mut self) {
        self.window_ms
            .read_command(&mut self.command_readers.set_window_ms);
        self.interval_ms
            .read_command(&mut self.command_readers.set_interval_ms);
        self.mix.read_command(&mut self.command_readers.set_mix);
        if let Some(mode) = self.command_readers.set_mode.read() {
            self.mode = mode;
        }
    }

    fn process(&mut self, input: &mut [Frame], dt: f64, info: &Info) {
        self.window_ms.update(dt * input.len() as f64, info);
        self.interval_ms.update(dt * input.len() as f64, info);
        self.mix.update(dt * input.len() as f64, info);
        let num_frames = input.len();
        let cap = self.cap_l.len();

        if self.mode != self.last_mode {
            // A mode change always re-arms from Passing: Freeze re-captures
            // its window fresh, Stutter re-schedules its next decision, Off
            // simply stops reading the buffer.
            self.state = State::Passing;
            self.since_decision = 0;
            self.last_mode = self.mode;
        }

        for (i, frame) in input.iter_mut().enumerate() {
            let t = (i + 1) as f64 / num_frames as f64;
            let mix = self.mix.interpolated_value(t).0.clamp(0.0, 1.0);

            // Always keep the capture buffer warm (cheap, no allocation).
            self.cap_l[self.write_pos] = frame.left;
            self.cap_r[self.write_pos] = frame.right;
            self.write_pos = (self.write_pos + 1) % cap;

            if self.mode == HoldMode::Off || mix <= 0.0 {
                self.state = State::Passing;
                continue;
            }

            let window_ms = self
                .window_ms
                .interpolated_value(t)
                .clamp(1.0, MAX_WINDOW_MS);
            let mut state = std::mem::take(&mut self.state);

            let wet = match self.mode {
                HoldMode::Off => unreachable!("handled above"),
                HoldMode::Freeze => {
                    let (start, len, mut offset) = match state {
                        State::Frozen { start, len, offset } => (start, len, offset),
                        _ => {
                            let len = (((window_ms / 1000.0) * self.sample_rate) as usize)
                                .clamp(1, cap - 1);
                            let start = (self.write_pos + cap - len) % cap;
                            (start, len, 0)
                        }
                    };
                    let w = Frame::new(
                        self.cap_l[(start + offset) % cap],
                        self.cap_r[(start + offset) % cap],
                    );
                    offset = (offset + 1) % len;
                    state = State::Frozen { start, len, offset };
                    w
                }
                HoldMode::Stutter => match state {
                    State::Passing => {
                        self.since_decision += 1;
                        let mut next = State::Passing;
                        if self.since_decision >= self.next_decision_in {
                            self.since_decision = 0;
                            let interval_ms = self.interval_ms.interpolated_value(t).max(20.0);
                            let jitter = 0.5 + self.roll();
                            self.next_decision_in =
                                ((interval_ms / 1000.0) * self.sample_rate * f64::from(jitter))
                                    .max(1.0) as u64;
                            if self.roll() < mix {
                                let len_frac = 0.4 + 0.6 * self.roll();
                                let len = (((window_ms / 1000.0)
                                    * self.sample_rate
                                    * f64::from(len_frac))
                                    as usize)
                                    .clamp(1, cap - 1);
                                let start = (self.write_pos + cap - len) % cap;
                                next = State::Repeating {
                                    read_pos: start,
                                    len,
                                    remaining: len,
                                };
                            }
                        }
                        state = next;
                        *frame
                    }
                    State::Repeating {
                        mut read_pos,
                        len,
                        mut remaining,
                    } => {
                        let w = Frame::new(self.cap_l[read_pos], self.cap_r[read_pos]);
                        read_pos = (read_pos + 1) % cap;
                        remaining -= 1;
                        if remaining == 0 {
                            self.since_decision = 0;
                            state = State::Passing;
                        } else {
                            state = State::Repeating {
                                read_pos,
                                len,
                                remaining,
                            };
                        }
                        w
                    }
                    State::Frozen { .. } => {
                        // Switched modes without an intervening `process` —
                        // treat as freshly passing.
                        state = State::Passing;
                        *frame
                    }
                },
            };
            self.state = state;
            *frame = wet * mix.sqrt() + *frame * (1.0 - mix).sqrt();
        }
    }
}

/// Controls a [`Hold`] effect.
#[derive(Debug)]
pub struct HoldHandle {
    command_writers: CommandWriters,
}

impl HoldHandle {
    /// Sets the operating mode. Always re-arms from `Passing`: `Freeze`
    /// re-captures its window fresh, `Stutter` re-schedules its next decision.
    pub fn set_mode(&mut self, mode: HoldMode) {
        self.command_writers.set_mode.write(mode);
    }

    /// Sets the capture window length, in milliseconds.
    pub fn set_window_ms(&mut self, window_ms: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers
            .set_window_ms
            .write(ValueChangeCommand {
                target: window_ms.into(),
                tween,
            });
    }

    /// Sets the average time between stutter decisions, in milliseconds.
    pub fn set_interval_ms(&mut self, interval_ms: impl Into<Value<f64>>, tween: Tween) {
        self.command_writers
            .set_interval_ms
            .write(ValueChangeCommand {
                target: interval_ms.into(),
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
    set_mode: HoldMode,
    set_window_ms: ValueChangeCommand<f64>,
    set_interval_ms: ValueChangeCommand<f64>,
    set_mix: ValueChangeCommand<Mix>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::info::MockInfoBuilder;

    fn run(builder: HoldBuilder, input: &[f32]) -> Vec<Frame> {
        let (mut effect, _handle) = builder.build();
        effect.init(44_100, input.len().max(1));
        let mut frames: Vec<Frame> = input.iter().map(|&x| Frame::from_mono(x)).collect();
        let info = MockInfoBuilder::new().build();
        effect.process(&mut frames, 1.0 / 44_100.0, &info);
        frames
    }

    #[test]
    fn neutral_at_mode_off_is_bit_exact() {
        let input: Vec<f32> = (0..44_100).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
        let out = run(HoldBuilder::new().mix(Mix::WET), &input);
        for (a, b) in input.iter().zip(out.iter()) {
            assert_eq!(*a, b.left);
        }
    }

    #[test]
    fn neutral_at_zero_mix_is_bit_exact_even_in_freeze() {
        let input: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
        let out = run(
            HoldBuilder::new().mode(HoldMode::Freeze).window_ms(100.0),
            &input,
        );
        for (a, b) in input.iter().zip(out.iter()) {
            assert_eq!(*a, b.left);
        }
    }

    #[test]
    fn freeze_loops_a_captured_window_with_high_autocorrelation_at_the_lag() {
        // White-noise-like seeded input, so a genuine loop (not a coincidence)
        // is what produces near-1.0 autocorrelation at exactly the window lag.
        let mut lcg = Lcg(1234);
        let input: Vec<f32> = (0..44_100).map(|_| noise(&mut lcg) * 0.5).collect();
        let out = run(
            HoldBuilder::new()
                .mode(HoldMode::Freeze)
                .window_ms(50.0)
                .mix(Mix::WET),
            &input,
        );
        let lag = (0.050 * 44_100.0) as usize;
        // Compare well after the freeze has captured (skip the first window).
        let start = lag * 3;
        let n = 4000;
        let a: Vec<f32> = out[start..start + n].iter().map(|f| f.left).collect();
        let b: Vec<f32> = out[start + lag..start + lag + n]
            .iter()
            .map(|f| f.left)
            .collect();
        let corr = crate::dsp::correlation(&a, &b);
        assert!(
            corr > 0.99,
            "freeze did not loop cleanly at the window lag: corr={corr}"
        );
    }

    #[test]
    fn stutter_eventually_repeats_a_slice_on_a_monotonic_ramp() {
        let input: Vec<f32> = (0..44_100).map(|i| i as f32 / 44_100.0).collect();
        let out = run(
            HoldBuilder::new()
                .mode(HoldMode::Stutter)
                .interval_ms(15.0)
                .window_ms(40.0)
                .mix(Mix::WET)
                .seed(11),
            &input,
        );
        let mut saw_repeat = false;
        for w in out.windows(50) {
            if w[0].left > w[49].left + 1e-6 {
                saw_repeat = true;
                break;
            }
        }
        assert!(
            saw_repeat,
            "stutter never repeated a slice on a monotonic ramp"
        );
    }

    #[test]
    fn deterministic_given_the_same_seed() {
        let input: Vec<f32> = (0..20_000)
            .map(|i| (i as f32 * 0.015).sin() * 0.6)
            .collect();
        let a = run(
            HoldBuilder::new()
                .mode(HoldMode::Stutter)
                .mix(0.6)
                .interval_ms(25.0)
                .seed(42),
            &input,
        );
        let b = run(
            HoldBuilder::new()
                .mode(HoldMode::Stutter)
                .mix(0.6)
                .interval_ms(25.0)
                .seed(42),
            &input,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn no_allocation_shaped_process_never_panics_or_nans() {
        for block in [1usize, 5, 128, 4000] {
            let input = vec![0.4f32; block];
            for mode in [HoldMode::Off, HoldMode::Stutter, HoldMode::Freeze] {
                let out = run(
                    HoldBuilder::new()
                        .mode(mode)
                        .mix(0.9)
                        .interval_ms(5.0)
                        .window_ms(200.0),
                    &input,
                );
                for f in &out {
                    assert!(f.left.is_finite());
                }
            }
        }
    }
}
