// Test-module assert_eq! on f32 health values: intentional exact-equal check
// (serde round-trip must preserve the f32 bit-exactly).
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `ts_net` — shared networking contract for the TERMINAL STATE multiplayer testbed.
//!
//! This crate owns:
//! - Wire types: [`types::NetPlayer`], [`types::Bullet`] (replicated ECS components)
//! - Client-to-server messages: [`events::MoveIntent`], [`events::FireInput`]
//! - [`NetPlugin`] — registers replication rules + client messages via `bevy_replicon`
//!
//! **No terminal, no audio, no rendering** — pure ECS data + plugin registration.
//! Only `ts_netplay` depends on terminal/render crates.
//!
//! This is a testbed crate: it exists to evaluate `bevy_replicon` replication,
//! not to ship networked play in the main game. It is excluded from the game
//! build (see the `no_net_in_game_build` guard test in `ts_netplay`).
//!
//! # Usage
//! `NetPlugin` is shared by BOTH the server and client apps. Add it together with
//! `bevy_replicon`'s `RepliconPlugins` and the renet backend's
//! `RepliconRenetPlugins`, then create the renet server/client resources (see
//! `ts_netplay`):
//! ```ignore
//! app.add_plugins((MinimalPlugins, StatesPlugin, RepliconPlugins, RepliconRenetPlugins, NetPlugin));
//! ```

use bevy::prelude::*;
use bevy_replicon::prelude::{AppRuleExt, Channel, ClientMessageAppExt};

pub mod events;
pub mod types;

/// The shared networking plugin. Add this to both the server App and the client
/// App, AFTER `RepliconPlugins` (it registers replication rules + client messages
/// which need the replicon registries to already exist).
///
/// It registers:
/// - replication for [`types::NetPlayer`] and [`types::Bullet`] (server→client),
/// - the client→server messages [`events::MoveIntent`] and [`events::FireInput`]
///   on the ordered channel.
///
/// It deliberately does NOT add `RepliconPlugins` itself: the channel set must be
/// frozen only AFTER every message/rule is registered (the renet backend reads
/// `RepliconChannels` to size its renet channels), and the caller controls the
/// `RepliconPlugins` configuration (e.g. `ServerPlugin::tick_schedule`). Adding
/// `RepliconPlugins` here would force that choice on the caller.
pub struct NetPlugin;

impl Plugin for NetPlugin {
    fn build(&self, app: &mut App) {
        app
            // Server-authoritative replicated state.
            .replicate::<types::NetPlayer>()
            .replicate::<types::Bullet>()
            // Client→server intents (ordered so the latest move/fire is coherent).
            .add_client_message::<events::MoveIntent>(Channel::Ordered)
            .add_client_message::<events::FireInput>(Channel::Ordered);
    }
}
