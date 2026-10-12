//! `field_fx` — custom kira [`Effect`](kira::effect::Effect) implementations for
//! sound fields that kira has **no built-in equivalent** for. Every stage kira
//! already ships (EQ, filter, distortion, delay, reverb) is used natively from
//! the field insert chain `playback.rs` builds instead of being reinvented here;
//! this module holds only the handful of stages kira genuinely lacks.
//!
//! Each effect lives in its own submodule (mirrors kira's own
//! `effect/<name>/{builder,handle}.rs` layout so `kira::command_writers_and_readers!`
//! — which declares module-scoped `CommandWriters`/`CommandReaders` types — never
//! collides across effects):
//!
//! * [`bitcrusher`] — bit-depth quantization + sample & hold decimation.
//!   Kira's [`DistortionBuilder`](kira::effect::distortion::DistortionBuilder) is
//!   waveshaping (hard/soft clip); it has no quantization or decimation stage.
//! * [`ring_mod`] — amplitude modulation by an internal sine carrier. Kira has
//!   no oscillator/modulation effect (its `modulator` module drives *parameters*,
//!   not audio-rate multiplication of the signal itself).
//! * [`wow_flutter`] — a short, continuously-modulated fractional delay
//!   (combined slow "wow" + fast "flutter" LFOs) that reads back the recent
//!   signal at a drifting lag, producing pitch wobble. Kira's
//!   [`DelayBuilder`](kira::effect::delay::DelayBuilder) fixes `delay_time` at
//!   construction with no live tween and no fractional/modulated read.
//! * [`glitch_stutter`] (module name kept; [`crate::field`] calls the effect
//!   `Hold`) — deterministic seeded buffer-repeat (stutter) or a sustained
//!   captured-window loop (freeze). Nothing like it exists in kira.
//! * [`stereo_width`] (`Width`) — mid/side collapse-to-mono / widen. Kira's
//!   reverb has a `stereo_width` knob but only for its own wet signal; there
//!   is no general-purpose stereo-image effect.
//! * [`tremolo_gate`] — LFO amplitude modulation (smooth tremolo or a hard
//!   on/off gate). Kira has no tremolo/gate effect. Not part of the five-effect
//!   field chain [`crate::field`] drives (`Crush`/`RingMod`/`Wow`/`Hold`/`Width`)
//!   — kept as real, tested, standalone code for hosts that want a plain
//!   tremolo/gate outside the field system.
//!
//! Every effect here is **real-time-safe** (allocation only in `init`/
//! `on_change_sample_rate`, never in `process`), **deterministic** (state
//! advances from a per-sample counter and a seeded LCG — never the wall clock,
//! never thread-local anything), and **click-free at weight 0** (every
//! continuous parameter is a kira [`Parameter`](kira::Parameter) driven by the
//! same tween machinery kira's own effects use, and the effect's own "amount"
//! parameter is wired so that its 0 value reproduces the dry input exactly, not
//! approximately — see each submodule's neutral-at-zero note).
//!
//! "Ducking" (sidechain-like level reduction) is deliberately **not** a custom
//! effect here: kira already exposes exactly the primitive it needs — a mixer
//! track's own `volume` [`Parameter<Decibels>`](kira::Decibels), tweenable with
//! no clicks. Each field bus `playback.rs` builds is a dedicated, otherwise
//! untouched track whose volume [`crate::field::FieldMix`]'s `vol_duck_db`/
//! `pump_depth_db` drive directly, so "ducking" costs zero extra DSP.

pub mod bitcrusher;
pub mod glitch_stutter;
pub mod ring_mod;
pub mod stereo_width;
pub mod tremolo_gate;
pub mod wow_flutter;
