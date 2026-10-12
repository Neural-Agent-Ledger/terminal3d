//! Wire types replicated between server and clients.
//!
//! These are the server-authoritative components `bevy_replicon` keeps in sync.
//! Both are plain `#[derive(Component, Serialize, Deserialize, Clone)]` data —
//! NO terminal, NO render. The server owns the authoritative copy; clients
//! receive replicated copies and render them.
//!
//! # Owner identity (why `u64`, not `Entity`)
//! [`Bullet::owner`] and [`NetPlayer::owner`] carry a plain `u64` **owner id**,
//! NOT a Bevy [`Entity`]. Replicating an `Entity` field requires replicon's
//! `#[entities]` / `MapEntities` machinery so the server entity is mapped to the
//! client's local entity on receive. We deliberately sidestep that: the server
//! assigns each connection a small monotonic `owner_id` (see
//! `ts_netplay`'s `OwnerIds` map) and stamps it onto that client's `NetPlayer`
//! and every `Bullet` it fires. A `u64` serializes trivially and means the same
//! thing on every peer, so a client can tell "this is MY player / MY bullet"
//! without any entity mapping. The spec (§1, "store owner as a client id u64 to
//! avoid entity-mapping complexity — your call, document it") explicitly allows
//! this; this is the chosen approach.

use bevy::prelude::*;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Per-player replicated state. Owned by the server; replicated to all clients.
///
/// The server moves this in response to that client's [`MoveIntent`](crate::events::MoveIntent)
/// (collision-checked against the arena), and decrements [`health`](Self::health)
/// when a bullet hits. Clients read every replicated `NetPlayer` to draw the
/// other players (and to know their own authoritative pose / health).
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct NetPlayer {
    /// World position in cell units (same convention as [`re_world::CameraPose::pos`]).
    pub pos: Vec3,
    /// Facing yaw in radians (`dir = (sin yaw, cos yaw)`, matching `CameraPose`).
    pub yaw: f32,
    /// Remaining health in `[0, MAX_HEALTH]`. Reaches `0` on enough hits, then the
    /// server respawns the player (resetting pos + health).
    pub health: f32,
    /// The connection's stable owner id (see the module docs). Lets a client
    /// identify which replicated `NetPlayer` is its own.
    pub owner: u64,
}

impl NetPlayer {
    /// Full health a freshly spawned / respawned player starts with.
    pub const MAX_HEALTH: f32 = 100.0;

    /// A new full-health player at `pos` facing `yaw`, tagged with `owner`.
    pub fn new(pos: Vec3, yaw: f32, owner: u64) -> Self {
        NetPlayer {
            pos,
            yaw,
            health: Self::MAX_HEALTH,
            owner,
        }
    }
}

/// A server-spawned, server-integrated projectile. Replicated so every client
/// can render it as a distance-accurate ASCII point-sprite
/// (`re_render::bullet_sprite::splat_bullets`).
///
/// The server integrates `pos += vel * dt` each tick, tests world-solid and
/// player-sphere collisions, applies damage, and despawns the bullet on hit,
/// world impact, or [`ttl`](Self::ttl) expiry.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Bullet {
    /// World position in cell units.
    pub pos: Vec3,
    /// Velocity in cells/second (direction × muzzle speed).
    pub vel: Vec3,
    /// Owner id of the shooter (see [`NetPlayer::owner`]). A bullet never damages
    /// its own owner.
    pub owner: u64,
    /// Seconds of life remaining; the server despawns the bullet at `ttl <= 0`.
    pub ttl: f32,
}

impl Bullet {
    /// Spawn a bullet at `pos` travelling along `vel`, owned by `owner`, living
    /// for `ttl` seconds.
    pub fn new(pos: Vec3, vel: Vec3, owner: u64, ttl: f32) -> Self {
        Bullet {
            pos,
            vel,
            owner,
            ttl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn netplayer_roundtrips_through_serde() {
        let p = NetPlayer::new(Vec3::new(1.0, 2.0, 0.5), 1.5, 7);
        let bytes = postcard::to_allocvec(&p).expect("serialize NetPlayer");
        let back: NetPlayer = postcard::from_bytes(&bytes).expect("deserialize NetPlayer");
        assert_eq!(p, back);
        assert_eq!(back.health, NetPlayer::MAX_HEALTH);
    }

    #[test]
    fn bullet_roundtrips_through_serde() {
        let b = Bullet::new(Vec3::new(3.0, 4.0, 0.5), Vec3::new(0.0, 9.0, 0.0), 2, 3.0);
        let bytes = postcard::to_allocvec(&b).expect("serialize Bullet");
        let back: Bullet = postcard::from_bytes(&bytes).expect("deserialize Bullet");
        assert_eq!(b, back);
    }
}
