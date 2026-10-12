//! Local-player client-side prediction reconciliation.
//!
//! ## Prediction already exists
//!
//! `re_present`'s terminal backend already does the *prediction* half of
//! client-side prediction: its `held_move_sys` moves the local `CameraPose`
//! entity from the held keys every frame (via `move_with_collision`), so the
//! local view responds instantly to input without waiting for the server
//! round-trip. We do NOT add a second mover — prediction = that existing
//! movement.
//!
//! ## What this module adds: reconciliation
//!
//! Because the server is authoritative, the predicted local pose can drift from
//! the server's truth (collision resolved differently, packets dropped, etc.).
//! Reconciliation corrects that drift against the replicated, authoritative
//! `NetPlayer` for the local owner. This testbed **snaps** on divergence rather
//! than replaying buffered inputs — a snap is simpler, deterministic, and at LAN
//! latencies imperceptible. The input history is still recorded ([`InputHistory`])
//! so the design is complete and a future replay-based reconciler has the data.
//!
//! Two deliberate restrictions, documented inline:
//! - We reconcile **only x and y**. The server only moves the player on the
//!   ground plane (`apply_move` touches x/y); z is the client's local gravity in
//!   `re_present`. Snapping z would fight that local fall, so z is left alone.
//! - We identify the local authoritative player as the replicated `NetPlayer`
//!   nearest the predicted camera, locked once chosen (see [`LocalOwner`]).
//!
//! Pure compute: NO terminal, NO transport. [`reconcile_pose`] is the
//! unit-test target.

use bevy::prelude::*;
use glam::Vec3;
use re_world::CameraPose;
use ts_net::events::MoveIntent;
use ts_net::types::NetPlayer;

/// Position divergence (cells) beyond which the local pose is snapped to the
/// server. Below this the prediction is "close enough" and left untouched to
/// avoid visible micro-corrections every frame.
pub const RECONCILE_POS_EPS: f32 = 0.05;

/// Yaw divergence (radians, shortest-arc) beyond which yaw is snapped.
pub const RECONCILE_YAW_EPS: f32 = 0.02;

/// Maximum number of sent intents retained in [`InputHistory`].
pub const INPUT_HISTORY_CAP: usize = 64;

/// Ring buffer of the `MoveIntent`s this client has sent, stamped with the
/// client-local clock.
///
/// We SNAP rather than replay, so this history is not consumed by the current
/// reconciler — but recording it keeps the prediction design complete and gives
/// a future replay-based reconciler (re-apply unacked inputs after a server
/// correction) the data it needs without a wire change.
#[derive(Resource, Default)]
pub struct InputHistory {
    /// `(client-local timestamp, intent)` pairs, oldest first.
    pub sent: std::collections::VecDeque<(f32, MoveIntent)>,
}

impl InputHistory {
    /// Record a sent intent, evicting the oldest beyond [`INPUT_HISTORY_CAP`].
    pub fn record(&mut self, t: f32, intent: MoveIntent) {
        self.sent.push_back((t, intent));
        while self.sent.len() > INPUT_HISTORY_CAP {
            self.sent.pop_front();
        }
    }
}

/// The replicated `NetPlayer` entity treated as the LOCAL authoritative player.
///
/// The client cannot learn its server-assigned `owner` id without a wire change
/// (the testbed sends no client→owner mapping back), so we cannot match the
/// local camera to its `NetPlayer` by id. The heuristic used here — the testbed
/// choice, with its limitation documented — is: pick the replicated `NetPlayer`
/// ENTITY nearest the predicted camera (x/y distance) and LOCK it, only
/// re-picking if that entity disappears. Locking avoids frame-to-frame
/// flip-flop when two players are momentarily close.
///
/// Limitation: with players spawned on top of each other this can lock onto the
/// wrong one until they separate. The clean fix is an additive wire field
/// carrying the server-assigned owner id back to its own client.
#[derive(Resource, Default)]
pub struct LocalOwner(
    /// The locked-on replicated `NetPlayer` entity, or `None` before the
    /// first pick.
    pub Option<Entity>,
);

/// Reconcile a predicted pose against the server-authoritative pose.
///
/// Snaps x/y and yaw to the server values when divergence exceeds the epsilons,
/// returning `true` if a correction was applied. **z is left untouched** (it is
/// the client's local gravity, which the server does not simulate) and the
/// approach is **snap, not replay**.
///
/// - position divergence is measured on the ground plane only (x, y);
/// - yaw divergence is the shortest-arc absolute angle.
pub fn reconcile_pose(
    predicted: &mut CameraPose,
    server_pos: Vec3,
    server_yaw: f32,
    pos_eps: f32,
    yaw_eps: f32,
) -> bool {
    let dx = server_pos.x - predicted.pos.x;
    let dy = server_pos.y - predicted.pos.y;
    let pos_div = (dx * dx + dy * dy).sqrt();

    // Shortest-arc yaw difference magnitude.
    let mut yd = (server_yaw - predicted.yaw) % std::f32::consts::TAU;
    if yd > std::f32::consts::PI {
        yd -= std::f32::consts::TAU;
    } else if yd < -std::f32::consts::PI {
        yd += std::f32::consts::TAU;
    }
    let yaw_div = yd.abs();

    if pos_div > pos_eps || yaw_div > yaw_eps {
        // Snap x/y to the server; leave z to local gravity. Snap yaw too.
        predicted.pos.x = server_pos.x;
        predicted.pos.y = server_pos.y;
        predicted.yaw = server_yaw;
        true
    } else {
        false
    }
}

/// Snap the local camera to the authoritative `NetPlayer` when it has diverged.
///
/// Resolves the local owner via [`LocalOwner`] (locked nearest-`NetPlayer`
/// heuristic) and applies [`reconcile_pose`]. Uses no `Time`, so it is cheap to
/// drive from a test with a single `app.update()`.
pub fn reconcile_local_player(
    mut local: ResMut<LocalOwner>,
    players: Query<(Entity, &NetPlayer)>,
    mut cam_q: Query<&mut CameraPose>,
) {
    let Ok(mut cam) = cam_q.single_mut() else {
        return; // no local camera yet
    };

    // Resolve the locked local-owner entity, or pick the nearest player.
    let owner_entity = match local.0 {
        Some(e) if players.get(e).is_ok() => e,
        _ => {
            let mut best: Option<(Entity, f32)> = None;
            for (e, np) in &players {
                let dx = np.pos.x - cam.pos.x;
                let dy = np.pos.y - cam.pos.y;
                let d2 = dx * dx + dy * dy;
                if best.is_none_or(|(_, bd)| d2 < bd) {
                    best = Some((e, d2));
                }
            }
            let Some((e, _)) = best else {
                return; // no players to reconcile against
            };
            local.0 = Some(e);
            e
        }
    };

    let Ok((_, np)) = players.get(owner_entity) else {
        return;
    };
    reconcile_pose(
        &mut cam,
        np.pos,
        np.yaw,
        RECONCILE_POS_EPS,
        RECONCILE_YAW_EPS,
    );
}
