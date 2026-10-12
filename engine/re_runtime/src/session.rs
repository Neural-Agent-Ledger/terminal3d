//! `session` — the play-session seam: the two resources every runtime and
//! game system needs to find "the camera that is playing" and "the level that
//! is loaded", independent of the host's own state machine.

use bevy::prelude::*;
use re_level::Level;

/// Marker for the camera entity the flow spawns, so `OnExit(Playing)` can
/// despawn exactly it (and nothing else).
#[derive(Component)]
pub struct PlayCamera;

/// The parsed [`Level`] currently loaded into the play loop, kept as a resource
/// so a host's own text/story layer can read its node word-pools on solve (the
/// `Map` / `PuzzleState` resources drop the pools). `None` outside "playing".
/// The host inserts it on level enter and clears it on level exit.
#[derive(Resource, Clone, Debug, Default)]
pub struct CurrentLevelData(pub Option<Level>);
