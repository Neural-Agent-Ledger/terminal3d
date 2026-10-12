// Test-module assert_eq! on f32 values are intentional exact-equal checks
// (e.g. "clamped to exactly 0.0 / 1.0").  Allow float_cmp only in test
// compilation units; lib code is still subject to the deny.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_audio` — TERMINAL STATE's procedural rhythmic audio: an in-house synth + a
//! kira-backed playback/clock/effects layer + a generative arrangement layer.
//!
//! Layers (mirroring the rest of TERMINAL STATE's pure-compute vs. device split):
//! * [`synth`] — pure-maths synthesis: oscillators (sine/square/tri/saw/noise),
//!   an ADSR [`synth::Envelope`], and a deterministic LCG noise source driving a
//!   declarative recipe table that renders the game's sounds to mono f32 PCM at
//!   44.1 kHz. Engine-agnostic; determinism-tested (build-vs-build, not pinned
//!   to a golden baseline). ZERO device.
//! * [`transport`] — pure logic: the BPM tick grid, strong/weak beat classes,
//!   the snap-window [`transport::next_tick`], and humanize config. ZERO device.
//! * [`policy`] — pure table: [`policy::quantize_of`] maps each [`synth::SoundId`]
//!   to `Immediate` vs `Beat(Strong|Weak)`. Exhaustive. ZERO device.
//! * [`ambience`] — pure mood abstraction: [`ambience::AmbienceMood`] (the four
//!   drone beds as named moods), the intensity→dread curve
//!   ([`ambience::AmbienceTargets`]), and the equal-power [`ambience::Crossfade`]
//!   state machine the device layer drives between beds. ZERO device.
//! * [`director`] — pure logic with its own seeded LCG: generative arrangement
//!   (sparse tonal notes, drone accents, terminal glitches) per
//!   [`director::Theme`] — including the four campaign movements
//!   (`P1FirstLight`, `P2TheGaze`, `P3Grief`, `P4Closing`) and their motifs —
//!   plus the solve-silence window ([`Director::hold_breath`]) that mutes the
//!   theme layer while the ambience beds keep running. Reproducible. ZERO device.
//! * [`playback`] — the kira backend: `AudioManager` + master/accent effect
//!   tracks + a musical clock, converting [`synth`] PCM into kira
//!   `StaticSoundData`, scheduling onto clock ticks, and degrading to a silent
//!   no-op when no device exists. The ONLY module that touches a device.
//! * [`plugin`] — the Bevy [`plugin::AudioPlugin`]: a message-driven trigger
//!   surface ([`plugin::PlaySound`], [`plugin::SetAmbient`],
//!   [`plugin::SetMasterVolume`], [`plugin::SetTempo`], [`plugin::SetTheme`],
//!   [`plugin::SolveSilence`]) plus the systems that run the director and
//!   resolve scheduling.
//!
//! * [`cue`] — pure table: game events ([`cue::Cue`]) → voiced layers, the
//!   one place the sound GRAMMAR lives (rising = progress, falling = regress,
//!   consonant = correct, dissonant = blocked).
//! * [`acoustics`] — pure room model: a level's size / sky / absorption →
//!   RT60, image-source early reflections, and a deterministic stereo room
//!   processor that bakes the room into the ambience.
//! * [`spatial`] — pure positional audio: listener (camera pos + yaw) →
//!   distance-attenuated, yaw-panned, air-absorbed, occluded voices with a
//!   deterministic voice budget.
//! * [`local_room`] — pure per-POSITION room estimation from ray-probe
//!   distances (the game ray-marches the grid; this module turns the result
//!   into a [`Room`] and smooths it), so reverb changes with where the player
//!   is standing, not once per level.
//! * [`soundscape`] — pure stereo ambience: decorrelated drone bed, mid
//!   texture, progress-driven harmony voices, seeded detail events, positional
//!   loops.
//! * [`drone_music`] — the slower, longer-form evolving background MUSIC
//!   layer: layered detuned partials under a slow filter sweep, a sub layer,
//!   an airy noise layer, and a seeded [`drone_music::MusicSequencer`] that
//!   shifts the harmonic root on a multi-minute timescale.
//! * [`offline`] — device-free mixdown of all of the above to stereo PCM/WAV,
//!   for tests and for auditioning without playing anything.
//! * [`dsp`] — the small deterministic DSP primitives those are built from.
//! * [`field`] — pure "sound fields" mixing: the 15-archetype table and the
//!   stateless [`field::combine`] that folds
//!   weighted [`field::FieldInput`]s into a [`field::FieldMix`] target state
//!   for the field insert chain. ZERO device, ZERO kira dependency.
//! * [`field_fx`] — the five custom kira [`Effect`](kira::effect::Effect)
//!   implementations the field chain needs and kira has no native equivalent
//!   for (`Crush`, `RingMod`, `Wow`, `Hold`, `Width`).
//! * [`offline_backend`] — a device-free, real kira [`Backend`](kira::backend::Backend)
//!   for rendering the actual playback graph (not a hand-rolled approximation)
//!   to PCM for offline/CI verification.
//!
//! Only [`playback`] touches kira's audio device; everything else — including
//! [`field_fx`] and [`offline_backend`], which touch kira's *library* types but
//! never open a device — compiles and runs with no device. No terminal, no
//! render, no `unsafe`. Wiring into the host game is a separate step.

pub mod acoustics;
pub mod ambience;
pub mod cue;
pub mod director;
pub mod drone_music;
pub mod dsp;
pub mod field;
pub mod field_fx;
pub mod local_room;
pub mod offline;
pub mod offline_backend;
pub mod playback;
pub mod plugin;
pub mod policy;
pub mod soundscape;
pub mod spatial;
pub mod synth;
pub mod transport;

// Re-export the common surface so callers can `use re_audio::{…}`.
pub use acoustics::Room;
pub use ambience::{Ambience, AmbienceMood, AmbienceTargets, Crossfade};
pub use cue::{Cue, NoteVoice, Surface};
pub use director::{Director, DirectorConfig, DirectorEvent, Theme};
pub use drone_music::{MusicSequencer, MusicState};
pub use field::{combine, Archetype, BusFx, FieldInput, FieldMix, FieldSettings, SpaceFx};
pub use local_room::{RoomProbe, RoomTracker};
pub use playback::{LoopHandle, Player};
pub use plugin::{
    AudioListener, AudioPlugin, AudioSources, HarmonyProgress, HarmonyTension, MirrorFootstep,
    PlayCue, PlaySound, PlayVoice, SetAmbience, SetAmbient, SetMasterVolume, SetRoom, SetTempo,
    SetTheme, SolveSilence, SoundFieldMix,
};
pub use policy::{quantize_of, Quantize};
pub use spatial::{Listener, LoopKind, PositionalSource, SourceKey};
pub use synth::{Envelope, SoundId, SynthBank, SynthSound, Wave, SAMPLE_RATE, SYNTH_SOUND_COUNT};
pub use transport::{next_tick, BeatClass, Grid, TickPos, Transport};
