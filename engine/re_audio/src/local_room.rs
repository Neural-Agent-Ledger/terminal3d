//! `local_room` — estimate a *local* acoustic [`Room`] from ray-probe
//! distances to nearby walls/ceiling/floor. **Pure logic, no device, no
//! world/grid access.**
//!
//! [`crate::acoustics::Room`] used to be one box per LEVEL (the whole voxel
//! extent). This module turns it into one box per POSITION: a player
//! standing in a doorway between a corridor and a hall hears a different
//! room than one standing deep in either, and a closet sounds like a closet
//! even inside a level that is mostly a big hall.
//!
//! The ray-marching itself needs the level grid (`re_world::Map`), which
//! `re_audio` must not depend on, so the boundary is: the host game walks the
//! grid and hands this module the resulting distances as a [`RoomProbe`];
//! everything from there is pure and unit-tested with no world at all.
//!
//! * [`RoomProbe`] — six ray distances (±x, ±y, ±z) from the listener to the
//!   first solid cell (capped at [`MAX_PROBE_CELLS`]), whether the upward ray
//!   escaped to open sky, and a material-absorption estimate.
//! * [`estimate_room`] — turns a probe into a [`Room`]: the box a listener
//!   standing in the middle of the given clearances would be inside.
//! * [`RoomTracker`] — a stateful smoother: quantizes the estimated room to
//!   coarse buckets and requires a minimum elapsed time AND a real change
//!   before emitting a new room, so looking around / a half-step does not
//!   retrigger a scene re-bake every frame — only a real corridor→hall
//!   transition does. [`crate::playback::Player::set_room`] already
//!   crossfades smoothly (2.5 s scene fade + an 800 ms reverb tween) once fed
//!   a new room; this module decides *when* to feed it one.

use crate::acoustics::Room;

/// Rays are capped here (cells) — beyond this the space is "big" for
/// acoustic purposes; probing further just wastes ray-march steps for no
/// audible difference (a 40-cell-wide hall and a 200-cell one both just sound
/// "large").
pub const MAX_PROBE_CELLS: f32 = 40.0;
/// A listener is never modeled as occupying less than this many cells of
/// clearance in each direction — avoids a degenerate zero-size room when a
/// probe ray is cast from flush against a wall.
const MIN_HALF_CELLS: f32 = 0.6;

/// Six ray-probe distances (cells) from a listener position, plus sky and
/// material information. [`estimate_room`] converts cells to metres via
/// [`crate::acoustics::METERS_PER_CELL`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomProbe {
    /// Distance to the first solid cell in `+x`.
    pub east: f32,
    /// Distance to the first solid cell in `-x`.
    pub west: f32,
    /// Distance to the first solid cell in `+y`.
    pub north: f32,
    /// Distance to the first solid cell in `-y`.
    pub south: f32,
    /// Distance to the first solid cell (or the map top) in `+z`.
    pub up: f32,
    /// Distance to the first solid cell (or the map bottom) in `-z`.
    pub down: f32,
    /// `true` when the upward ray left the map without hitting a solid cell
    /// AND the level is authored with a sky — i.e. the listener is standing
    /// somewhere with no ceiling overhead right now, even in a level that has
    /// indoor rooms elsewhere.
    pub open_sky: bool,
    /// Mean absorption coefficient of the surfaces the rays hit, in
    /// `[0.02, 0.95]` (see [`Room::absorption`]).
    pub absorption: f32,
}

impl RoomProbe {
    /// A probe with every ray at its cap and neutral absorption — the
    /// fallback for "no grid available" (e.g. before a map has loaded).
    pub fn open(absorption: f32) -> RoomProbe {
        RoomProbe {
            east: MAX_PROBE_CELLS,
            west: MAX_PROBE_CELLS,
            north: MAX_PROBE_CELLS,
            south: MAX_PROBE_CELLS,
            up: MAX_PROBE_CELLS,
            down: MAX_PROBE_CELLS,
            open_sky: true,
            absorption,
        }
    }

    /// Clamp every distance into `[0, MAX_PROBE_CELLS]` and absorption into
    /// its valid range. Ray-marchers hand back raw counts; this is the one
    /// place that makes a probe safe to turn into a [`Room`].
    pub fn clamped(self) -> RoomProbe {
        let c = |v: f32| v.clamp(0.0, MAX_PROBE_CELLS);
        RoomProbe {
            east: c(self.east),
            west: c(self.west),
            north: c(self.north),
            south: c(self.south),
            up: c(self.up),
            down: c(self.down),
            open_sky: self.open_sky,
            absorption: self.absorption.clamp(0.02, 0.95),
        }
    }
}

/// Estimate the local [`Room`] a listener at the probe's origin is standing
/// inside: width/depth/height from the paired ray distances (each ray at
/// least [`MIN_HALF_CELLS`], so a listener flush against a wall still gets a
/// sane box rather than a zero-width one), sky and absorption passed through.
pub fn estimate_room(probe: &RoomProbe) -> Room {
    let p = probe.clamped();
    let half = |v: f32| v.max(MIN_HALF_CELLS);
    let sx = half(p.east) + half(p.west);
    let sy = half(p.north) + half(p.south);
    let sz = half(p.up) + half(p.down);
    Room::from_cells(
        sx.round() as usize,
        sy.round() as usize,
        sz.round() as usize,
        p.open_sky,
        p.absorption,
    )
}

/// Coarse bucket size for quantizing a room's metres (one world cell) so
/// probe jitter — a half-step, a look around — does not compute as "a new
/// room" and spawn a scene re-render every frame.
const SIZE_BUCKET_M: f32 = crate::acoustics::METERS_PER_CELL;
/// Absorption quantization step.
const ABSORPTION_BUCKET: f32 = 0.04;

fn quantize(room: Room) -> Room {
    let q = |v: f32| (v / SIZE_BUCKET_M).round() * SIZE_BUCKET_M;
    Room {
        size_m: room.size_m.map(q),
        open_sky: room.open_sky,
        absorption: (room.absorption / ABSORPTION_BUCKET).round() * ABSORPTION_BUCKET,
    }
}

/// Smooths a stream of raw per-frame [`RoomProbe`] samples into occasional
/// [`Room`] updates: quantized to [`SIZE_BUCKET_M`]-metre buckets, rate
/// limited to at most one emission per [`RoomTracker::MIN_INTERVAL_S`], and
/// only emitted when the quantized room actually differs from the current
/// one. The caller (the host game) probes every play frame; this tracker
/// decides which of those probes are worth telling the device about.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomTracker {
    current: Room,
    since_last_s: f32,
}

impl RoomTracker {
    /// Minimum time between two emitted room changes (seconds). Bounds how
    /// often [`crate::playback::Player::set_room`] re-bakes the ambience
    /// scene as the player walks continuously — frequent enough that a real
    /// transition (corridor → hall) is felt within a stride or two, rare
    /// enough that it never spawns a render thread per frame.
    pub const MIN_INTERVAL_S: f32 = 0.35;

    /// Start tracking from an already-known room (e.g. the level's own
    /// whole-box estimate, until the first probe replaces it).
    pub fn new(initial: Room) -> RoomTracker {
        RoomTracker {
            current: quantize(initial),
            since_last_s: RoomTracker::MIN_INTERVAL_S,
        }
    }

    /// The room the tracker currently considers active (already quantized).
    pub fn room(&self) -> Room {
        self.current
    }

    /// Advance by `dt` seconds with a fresh raw probe. Returns `Some(room)`
    /// exactly when it is time to push a new room to the device (a real
    /// quantized change AND the minimum interval has elapsed since the last
    /// emission); `None` otherwise, in which case [`RoomTracker::room`] is
    /// still the room to keep using.
    pub fn update(&mut self, probe: &RoomProbe, dt: f32) -> Option<Room> {
        self.since_last_s += dt.max(0.0);
        let target = quantize(estimate_room(probe));
        if target == self.current || self.since_last_s < RoomTracker::MIN_INTERVAL_S {
            return None;
        }
        self.current = target;
        self.since_last_s = 0.0;
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(
        east: f32,
        west: f32,
        north: f32,
        south: f32,
        up: f32,
        down: f32,
        open_sky: bool,
    ) -> RoomProbe {
        RoomProbe {
            east,
            west,
            north,
            south,
            up,
            down,
            open_sky,
            absorption: 0.2,
        }
    }

    #[test]
    fn a_closet_rings_shorter_than_a_hall_from_the_same_probe_shape() {
        let closet = estimate_room(&probe(1.0, 1.0, 1.0, 1.0, 1.5, 1.5, false));
        let hall = estimate_room(&probe(12.0, 12.0, 10.0, 10.0, 3.0, 1.5, false));
        assert!(
            hall.rt60() > closet.rt60() * 2.0,
            "hall {} vs closet {}",
            hall.rt60(),
            closet.rt60()
        );
    }

    #[test]
    fn a_corridor_is_narrow_but_can_still_be_long() {
        // Narrow in x, long in y: width should be much smaller than depth.
        let corridor = estimate_room(&probe(0.8, 0.8, 20.0, 20.0, 1.5, 1.5, false));
        assert!(
            corridor.size_m[0] < corridor.size_m[1] * 0.2,
            "{:?}",
            corridor.size_m
        );
    }

    #[test]
    fn outdoors_is_open_sky_and_quieter_than_the_same_box_indoors() {
        // Same probe geometry — only the sky flag differs — isolates the
        // "no ceiling to reflect from" effect from room size.
        let out = probe(15.0, 15.0, 15.0, 15.0, 3.0, 1.5, true);
        let indoor = probe(15.0, 15.0, 15.0, 15.0, 3.0, 1.5, false);
        let ro = estimate_room(&out);
        let ri = estimate_room(&indoor);
        assert!(ro.open_sky);
        assert!(!ri.open_sky);
        assert_eq!(ro.size_m, ri.size_m, "test premise: only sky should differ");
        assert!(
            ro.rt60() < ri.rt60(),
            "outdoors should ring less: {} vs {}",
            ro.rt60(),
            ri.rt60()
        );
    }

    #[test]
    fn a_probe_flush_against_a_wall_still_yields_a_sane_room() {
        let flush = estimate_room(&probe(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false));
        assert!(flush.size_m.iter().all(|&v| v > 0.5));
        assert!(flush.rt60() > 0.1 && flush.rt60().is_finite());
    }

    #[test]
    fn estimate_is_deterministic() {
        let p = probe(5.0, 6.0, 7.0, 8.0, 2.5, 1.5, false);
        assert_eq!(estimate_room(&p), estimate_room(&p));
    }

    #[test]
    fn tracker_ignores_tiny_jitter_but_catches_a_real_transition() {
        let closet = probe(1.0, 1.0, 1.0, 1.0, 1.5, 1.5, false);
        let hall = probe(14.0, 14.0, 12.0, 12.0, 3.0, 1.5, false);
        let mut t = RoomTracker::new(estimate_room(&closet));

        // A near-identical probe (sub-cell jitter): the quantized room is
        // unchanged, so it never emits, however much time has passed.
        let jitter = probe(1.05, 0.98, 1.02, 0.99, 1.48, 1.51, false);
        assert!(t.update(&jitter, 10.0).is_none());

        // A real transition: the tracker starts "ready" (ctor primes the
        // interval), so the first observation of a real change emits at once.
        let first = t
            .update(&hall, 0.01)
            .expect("first real change should emit");
        assert!(first.rt60() > estimate_room(&closet).rt60());
        assert_eq!(t.room(), first);

        // Immediately requesting a DIFFERENT room again, before the minimum
        // interval has elapsed: suppressed by the rate limit even though it
        // is a real change (back toward the closet).
        assert!(
            t.update(&closet, 0.01).is_none(),
            "rate limit should suppress a rapid follow-up"
        );
        assert_eq!(
            t.room(),
            first,
            "the suppressed room must not become current"
        );

        // The same still-pending change, now enough time has elapsed: it
        // finally emits.
        let second = t
            .update(&closet, RoomTracker::MIN_INTERVAL_S)
            .expect("delayed change should emit");
        assert_eq!(second, quantize(estimate_room(&closet)));
    }

    #[test]
    fn tracker_quantizes_so_sub_bucket_probes_never_emit() {
        let base = probe(10.0, 10.0, 10.0, 10.0, 2.5, 1.5, false);
        let mut t = RoomTracker::new(estimate_room(&base));
        // Wiggle within less than one cell on every ray, for a while: never
        // a new emission, however much time passes.
        let mut dt_total = 0.0f32;
        for i in 0..20 {
            let d = 0.3 * ((i % 3) as f32 - 1.0);
            let p = probe(10.0 + d, 10.0 - d, 10.0 + d, 10.0 - d, 2.5, 1.5, false);
            let emitted = t.update(&p, 0.5);
            dt_total += 0.5;
            assert!(
                emitted.is_none(),
                "sub-bucket jitter emitted at t={dt_total}"
            );
        }
    }

    #[test]
    fn open_probe_fallback_is_a_sane_large_outdoor_room() {
        let r = estimate_room(&RoomProbe::open(0.15));
        assert!(r.open_sky);
        assert!(r.size_m.iter().all(|&v| v > 10.0));
    }
}
