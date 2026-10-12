//! Remote-entity interpolation for the multiplayer testbed.
//!
//! Replicated state arrives in discrete, jittery updates (server ticks at
//! ~60 Hz, delivered over UDP with variable latency). Rendering the *latest*
//! replicated position directly produces visible stutter and teleporting. The
//! standard fix is **entity interpolation**: buffer a short history of received
//! snapshots and render each remote entity at a slightly delayed time
//! `render_t = now - INTERP_DELAY`, smoothly LERPing between the two snapshots
//! that bracket that time. The delay is the "rendering buffer" that absorbs
//! jitter — at the cost of showing remote entities a tick or two in the past.
//!
//! This module is pure compute: it owns the snapshot ring buffers
//! ([`PlayerInterp`], [`BulletInterp`]), a capture system
//! ([`capture_snapshots`]) that records `Changed<T>` replicated state stamped
//! with the client-local clock, and the pure sampling helper [`sample_at`] (the
//! unit-test target). NO terminal, NO transport.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

use bevy::prelude::*;
use glam::Vec3;
use ts_net::types::{Bullet, NetPlayer};

/// How far in the past remote entities are rendered, in seconds.
///
/// `2.0 / 60.0` ≈ 33.3 ms = **two server ticks** at the 60 Hz sim rate. One
/// tick of buffer is the bare minimum to always have a *newer* snapshot to
/// LERP toward; two ticks gives a margin so a single late/dropped packet still
/// leaves a bracketing pair, avoiding a fallback to "hold newest" (which looks
/// like a brief freeze). Larger delays smooth more jitter but make remote
/// entities feel laggier; two ticks is the usual sweet spot for a LAN testbed.
pub const INTERP_DELAY: f32 = 2.0 / 60.0;

/// How long (seconds) snapshots are retained in each ring buffer.
///
/// At 60 Hz a `0.5 s` window holds ~30 snapshots — far more than the **≥3**
/// needed to always bracket `render_t = now - INTERP_DELAY` (we need one
/// snapshot at/after `render_t` and one before it; three gives slack against
/// pruning races and bursty delivery). Older entries are pruned every frame so
/// the buffers stay bounded regardless of session length.
pub const BUFFER_WINDOW: f32 = 0.5;

/// A captured remote-player pose snapshot. `Copy` so buffer entries are cheap
/// and the [`Interpolatable`] impl can take/return by value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerSample {
    /// World-space position at capture time.
    pub pos: Vec3,
    /// Facing yaw (radians) at capture time.
    pub yaw: f32,
}

/// Values that can be smoothly interpolated between two snapshots.
///
/// `lerp_to(other, t)` returns the value `t` of the way from `self` to `other`
/// (`t = 0` → `self`, `t = 1` → `other`).
pub trait Interpolatable: Copy {
    /// Blend `self` toward `other` by fraction `t` (`0` = `self`, `1` = `other`).
    fn lerp_to(self, other: Self, t: f32) -> Self;
}

impl Interpolatable for Vec3 {
    fn lerp_to(self, other: Self, t: f32) -> Self {
        self.lerp(other, t)
    }
}

impl Interpolatable for PlayerSample {
    fn lerp_to(self, other: Self, t: f32) -> Self {
        // Position is a straight component-wise LERP.
        let pos = self.pos.lerp(other.pos, t);
        // Yaw is an angle: interpolate along the SHORTEST ARC so a turn across
        // the ±π wrap (e.g. +179° → −179°) rotates the short way (2°) instead
        // of spinning the long way (358°) that a naive scalar LERP would give.
        let mut d = (other.yaw - self.yaw) % TAU;
        if d > PI {
            d -= TAU;
        } else if d < -PI {
            d += TAU;
        }
        let yaw = self.yaw + d * t;
        PlayerSample { pos, yaw }
    }
}

/// Sample a time-stamped snapshot buffer at `render_t`.
///
/// The buffer must be sorted ascending by timestamp (the capture system pushes
/// in time order, so it always is). Behaviour:
/// - empty buffer → `None`;
/// - `render_t <= oldest` → hold the oldest snapshot;
/// - `render_t >= newest` → hold the newest snapshot (extrapolation is *not*
///   done — holding avoids overshoot artefacts on a stalled stream);
/// - otherwise → find the consecutive pair `(t0, v0), (t1, v1)` with
///   `t0 <= render_t <= t1` and LERP by `(render_t - t0) / (t1 - t0)`.
pub fn sample_at<V: Interpolatable>(buf: &[(f32, V)], render_t: f32) -> Option<V> {
    let first = buf.first()?;
    if render_t <= first.0 {
        return Some(first.1);
    }
    // `last()` is `Some` here because `first()` was `Some`.
    let last = buf[buf.len() - 1];
    if render_t >= last.0 {
        return Some(last.1);
    }
    // Find the bracketing pair. The buffer is small (~30 entries) so a linear
    // scan is cheaper than a binary search and trivially correct.
    for pair in buf.windows(2) {
        let (t0, v0) = pair[0];
        let (t1, v1) = pair[1];
        if t0 <= render_t && render_t <= t1 {
            let span = t1 - t0;
            let frac = if span <= 0.0 {
                0.0
            } else {
                (render_t - t0) / span
            };
            return Some(v0.lerp_to(v1, frac));
        }
    }
    // Unreachable for a sorted buffer given the clamps above, but holding the
    // newest is the safe fallback.
    Some(last.1)
}

/// Per-remote-player snapshot ring buffers, keyed by replicated entity.
#[derive(Resource, Default)]
pub struct PlayerInterp {
    /// Entity → ascending-by-time `(client_local_time, sample)` history.
    pub buffers: HashMap<Entity, Vec<(f32, PlayerSample)>>,
}

/// Per-bullet snapshot ring buffers, keyed by replicated entity.
#[derive(Resource, Default)]
pub struct BulletInterp {
    /// Entity → ascending-by-time `(client_local_time, pos)` history.
    pub buffers: HashMap<Entity, Vec<(f32, Vec3)>>,
}

/// Record newly-replicated player/bullet state into the interpolation buffers,
/// stamped with the client-local clock, and prune stale entries.
///
/// Runs in `Update`. `bevy_replicon` applies received component values in
/// `PreUpdate`, so by `Update` the `Changed<T>` filter sees exactly the
/// entities whose replicated state advanced this frame — which is precisely
/// what we want to stamp at "now".
///
/// Pruning: each buffer drops entries older than `now - BUFFER_WINDOW`. At
/// 60 Hz a 0.5 s window cannot empty an actively-updating buffer, so we keep
/// the simple rule "retain by window, then drop any buffer that became empty"
/// — an empty buffer just means that entity stopped updating long ago, and it
/// will repopulate on its next `Changed` event. Buffers for despawned entities
/// are removed by intersecting with the live entity set.
pub fn capture_snapshots(
    time: Res<Time>,
    changed_players: Query<(Entity, &NetPlayer), Changed<NetPlayer>>,
    changed_bullets: Query<(Entity, &Bullet), Changed<Bullet>>,
    all_players: Query<Entity, With<NetPlayer>>,
    all_bullets: Query<Entity, With<Bullet>>,
    mut pi: ResMut<PlayerInterp>,
    mut bi: ResMut<BulletInterp>,
) {
    let now = time.elapsed_secs();
    let cutoff = now - BUFFER_WINDOW;

    // 1. Append this frame's changed snapshots.
    for (e, np) in &changed_players {
        pi.buffers.entry(e).or_default().push((
            now,
            PlayerSample {
                pos: np.pos,
                yaw: np.yaw,
            },
        ));
    }
    for (e, b) in &changed_bullets {
        bi.buffers.entry(e).or_default().push((now, b.pos));
    }

    // 2. Prune entries older than the window; drop buffers that emptied.
    for buf in pi.buffers.values_mut() {
        buf.retain(|(t, _)| *t >= cutoff);
    }
    pi.buffers.retain(|_, buf| !buf.is_empty());
    for buf in bi.buffers.values_mut() {
        buf.retain(|(t, _)| *t >= cutoff);
    }
    bi.buffers.retain(|_, buf| !buf.is_empty());

    // 3. Drop buffers whose entity no longer exists (despawned/disconnected).
    let live_players: std::collections::HashSet<Entity> = all_players.iter().collect();
    pi.buffers.retain(|e, _| live_players.contains(e));
    let live_bullets: std::collections::HashSet<Entity> = all_bullets.iter().collect();
    bi.buffers.retain(|e, _| live_bullets.contains(e));
}

/// Read interpolated poses for every buffered remote player at `render_t`.
///
/// This is the exposed, testable consumer of [`PlayerInterp`]. The current
/// first-person view only *splats bullets* (remote players are not yet drawn as
/// sprites — that is the sprite work, out of scope here), but keeping a real
/// consumer means the remote-player interpolation path is exercised and
/// unit-testable rather than dead code: a future remote-avatar renderer reads
/// exactly this.
pub fn interpolated_players(pi: &PlayerInterp, render_t: f32) -> Vec<(Entity, PlayerSample)> {
    pi.buffers
        .iter()
        .filter_map(|(&e, buf)| sample_at(buf, render_t).map(|s| (e, s)))
        .collect()
}

/// Fill `out` with the interpolated world position of every buffered bullet at
/// `render_t`. Clears `out` first; pushes nothing for buffers that sample to
/// `None` (only possible for an empty buffer, which pruning removes). Reuses the
/// caller's scratch vector so there is no per-frame allocation beyond it.
pub fn interpolated_bullets(bi: &BulletInterp, render_t: f32, out: &mut Vec<Vec3>) {
    out.clear();
    for buf in bi.buffers.values() {
        if let Some(pos) = sample_at(buf, render_t) {
            out.push(pos);
        }
    }
}
