//! Client-to-server messages for the multiplayer testbed.
//!
//! Registered with `bevy_replicon`'s `add_client_message::<T>(Channel::Ordered)`
//! in [`super::NetPlugin`]. The client writes them with `MessageWriter<T>`; the
//! server reads them as `MessageReader<FromClient<T>>` (the wrapper carries the
//! sender's `ClientId`). Both are plain serde data — no terminal, no entities,
//! so no `MapEntities` machinery is needed.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Sent by a client each frame describing how it wants to move THIS tick.
///
/// The server applies it to that client's [`NetPlayer`](crate::types::NetPlayer)
/// using `re_world` movement + collision against the loaded arena `Map`. The axes
/// are normalized intents in `[-1, 1]`; the server owns the actual speed so a
/// client cannot move faster by inflating the values (it clamps).
///
/// `Message`, not an `Event`, so the server can drain the latest value per client
/// with a plain `MessageReader<FromClient<MoveIntent>>` each tick.
#[derive(Message, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MoveIntent {
    /// Forward / backward along facing, `-1..=1` (negative = backward).
    pub forward: f32,
    /// Strafe right / left, `-1..=1` (negative = left).
    pub strafe: f32,
    /// Yaw turn delta for this tick, in radians (signed). The server adds it to
    /// the player's yaw (it does not scale by dt — the client already paced it).
    pub turn: f32,
    /// Vertical climb / descend intent, `-1..=1`. Reserved for free-fly arenas;
    /// the server may ignore it on a grounded arena.
    pub climb: f32,
}

/// Sent by a client when the fire key is pressed. The server spawns a
/// [`Bullet`](crate::types::Bullet) at the firing player's muzzle along the
/// supplied aim.
///
/// Carries the aim explicitly (`yaw` + `pitch`, radians) rather than relying on
/// the server's last-known facing, so a shot fired the same frame the player
/// turned still goes where the player was looking. `dir()` reconstructs the unit
/// direction with the engine's `(sin yaw, cos yaw)` convention.
#[derive(Message, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FireInput {
    /// Aim yaw in radians (`dir.xy = (sin yaw, cos yaw)`).
    pub yaw: f32,
    /// Aim pitch in radians (positive = up). `dir.z = sin pitch`.
    pub pitch: f32,
}

impl FireInput {
    /// The unit aim direction, matching `re_world`'s `(sin yaw, cos yaw)` ground
    /// convention with a pitch tilt applied to z. A zero-pitch shot is purely
    /// horizontal.
    pub fn dir(self) -> glam::Vec3 {
        let (sy, cy) = (self.yaw.sin(), self.yaw.cos());
        let (sp, cp) = (self.pitch.sin(), self.pitch.cos());
        // Horizontal component scaled by cos(pitch); vertical = sin(pitch).
        glam::Vec3::new(sy * cp, cy * cp, sp).normalize_or_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_intent_roundtrips() {
        let m = MoveIntent {
            forward: 1.0,
            strafe: -1.0,
            turn: 0.05,
            climb: 0.0,
        };
        let bytes = postcard::to_allocvec(&m).expect("serialize");
        let back: MoveIntent = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(m, back);
    }

    #[test]
    fn fire_input_roundtrips_and_aims() {
        let f = FireInput {
            yaw: 0.0,
            pitch: 0.0,
        };
        let bytes = postcard::to_allocvec(&f).expect("serialize");
        let back: FireInput = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(f, back);
        // yaw=0 faces +y (north) with the engine convention.
        let d = f.dir();
        assert!((d.x).abs() < 1e-6);
        assert!((d.y - 1.0).abs() < 1e-6);
        assert!((d.z).abs() < 1e-6);
    }
}
