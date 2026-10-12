//! `re_runtime` — the generic `.lvl` play-session runtime.
//!
//! This crate is the ECS layer that brings a loaded [`re_level::Level`] to
//! life once a host decides a level is live: the puzzle→render state bus,
//! prop power, emitter parts, sound fields, zoom, colour drift, trigger
//! fields, the faced-switch HUD probe, and note placards. It sits above
//! `re_core`/`re_world`/`re_level`/`re_puzzle`/`re_render`/`re_audio`/
//! `re_present` in the engine layering and is itself consumed by a game
//! binary — it knows nothing about menus, saves, a campaign, or any host's
//! concrete state enum.
//!
//! # The host-state seam
//!
//! Every plugin here that needs to know "is a level live right now" is
//! generic over `S: bevy::state::state::States` instead of hard-coding a
//! `Playing` variant. A host supplies the one state value that means "live"
//! when it constructs the plugin:
//!
//! ```ignore
//! app.add_plugins(re_runtime::zoom::ZoomPlugin { live: MyGameState::Playing });
//! app.add_plugins(re_runtime::sound_fields::SoundFieldsPlugin { live: MyGameState::Playing });
//! ```
//!
//! There is no shared "generic Playing state" type — only a value of the
//! host's own enum. This crate's own tests use a local
//! `#[derive(States)] enum TestState { #[default] Idle, Live }` to prove the
//! pattern works for any state shape, not just one game's.
//!
//! Plugins that need no state at all — [`prop_power::PropPowerPlugin`],
//! [`emit_parts::EmitPartsPlugin`], [`state_bridge::StateBridgePlugin`],
//! [`rng::GameRngPlugin`] — register unconditionally; the host adds them once
//! and they run for the lifetime of the App.
//!
//! # Seam resources the host writes
//!
//! A few systems need live data this crate must not know the shape of (a
//! campaign manifest, a settings screen, a cue tracker). Those cross the seam
//! as plain resources the host fills every frame or on level load, never as a
//! host type this crate reads directly:
//!
//! | resource | who fills it | what for |
//! |---|---|---|
//! | [`sound_fields::FieldContext`] | the host's level-enter handler | movement phase, fog distance, ambience mood |
//! | [`sound_fields::FieldSignals`] | the host's cue tracker, every frame | read-notes bitmask, recovered-shard count |
//! | [`sound_fields::FieldSettingsRes`] | the host's settings screen | accessibility intensity + reduced-effects toggle |
//! | [`context_line::HudPalette`] | passed directly into the HUD draw calls | the host's own title/background colours |
//!
//! Each of these degrades to a neutral default when the host has not filled
//! it yet (no campaign level current, no cue tracker installed), so a host
//! that only wants part of this crate's functionality is never forced to
//! wire the rest.
//!
//! # Minimal example
//!
//! Wiring one level into a headless Bevy `App`, with the play-session
//! resources and one state-gated system registered:
//!
//! ```ignore
//! use bevy::prelude::*;
//! use re_runtime::{session::CurrentLevelData, zoom::ZoomPlugin};
//!
//! #[derive(States, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
//! enum MyState {
//!     #[default]
//!     Menu,
//!     Playing,
//! }
//!
//! let mut app = App::new();
//! app.add_plugins(MinimalPlugins);
//! app.add_plugins(bevy::state::app::StatesPlugin);
//! app.init_state::<MyState>();
//! app.init_resource::<CurrentLevelData>();
//! app.add_plugins(ZoomPlugin { live: MyState::Playing });
//! ```
//!
//! A real host also adds [`prop_power::PropPowerPlugin`],
//! [`emit_parts::EmitPartsPlugin`], [`state_bridge::StateBridgePlugin`], and
//! [`sound_fields::SoundFieldsPlugin`] the same way, plus its own systems that
//! fill the seam resources above.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]

pub mod asset_notice;
pub mod audio_world;
pub mod color_drift_sources;
pub mod context_line;
#[cfg(feature = "dev")]
pub mod dev;
pub mod emit_parts;
pub mod hud_probe;
pub mod notes;
pub mod prop_power;
pub mod rng;
pub mod session;
pub mod sound_fields;
pub mod state_bridge;
pub mod triggers;
pub mod unstuck;
pub mod zoom;
