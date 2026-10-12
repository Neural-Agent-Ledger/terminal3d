// Test-module assert_eq! on f32 values are intentional exact-equal checks.
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `ts_netplay` library surface — the testbed's reusable pieces.
//!
//! The `terminal-state-netplay` binary (`src/main.rs`) is a thin CLI + app
//! wiring over these modules; exposing them as a library lets the integration
//! tests in `tests/` drive the SERVER SIM and renet TRANSPORT directly (a
//! binary's modules are not reachable from an integration test).
//!
//! - [`sim`] — server-authoritative ECS systems (movement, fire, bullet
//!   integrate/hit). Pure compute; NO terminal, NO render, NO transport.
//! - [`transport`] — renet/UDP socket + transport setup (server + client).
//! - [`client`] — input → intents, render replicated world+bullets, present via
//!   `re_present` (the ONLY terminal touch point in the workspace's net stack).
//! - [`interp`] — remote-entity interpolation: snapshot ring buffers + the pure
//!   `sample_at` LERP helper, rendering remote bullets/players at `now -
//!   INTERP_DELAY` to absorb network jitter.
//! - [`predict`] — local-player client-side prediction reconciliation: input
//!   history + snapping the predicted camera to the authoritative `NetPlayer`
//!   on divergence (prediction itself lives in `re_present`).
//!
//! This is an experimental multiplayer testbed, not shipped in the main game
//! build: it exists to evaluate `bevy_replicon` + renet replication over the
//! engine's existing world/present layers, isolated behind its own crate and
//! binary (see the `no_net_in_game_build` guard test).

pub mod client;
pub mod interp;
pub mod predict;
pub mod sim;
pub mod transport;
