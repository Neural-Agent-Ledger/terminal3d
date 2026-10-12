//! `audio_world` — the ONE place that walks a [`re_world::Map`] for audio
//! purposes. `re_audio` is engine-agnostic and knows nothing about the voxel
//! grid; it only consumes the pure numbers
//! this module produces:
//!
//! * [`probe_room`] — cast a short ray-probe in each of the six axis
//!   directions from a world position to build the
//!   [`re_audio::local_room::RoomProbe`] that `re_audio::local_room::estimate_room`
//!   turns into a local [`re_audio::acoustics::Room`]. Called every play frame
//!   (throttled by `re_audio::local_room::RoomTracker`) so the room the player
//!   hears tracks *where they are standing*, not just which level is loaded —
//!   a corridor, the hall it opens into, and a closet off it are three
//!   different rooms even though they share one `.lvl` file.
//! * [`occlusion_between`] — sample the straight line between a source and
//!   the listener for solid geometry, so a lamp on the other side of a wall
//!   sounds muffled and quieter (`re_audio::spatial::PositionalSource::occlusion`)
//!   instead of identical to one in plain view.
//!
//! No seams: it walks the generic [`re_world::Map`] only, never a game type.
//! [`probe_room`] and [`occlusion_between`] are `pub` because they cross a
//! crate boundary: [`crate::sound_fields`] calls `occlusion_between` in this
//! crate, and a host's own audio-cue code can call either directly.

use re_audio::local_room::{RoomProbe, MAX_PROBE_CELLS};
use re_world::Map;

/// March one axis-aligned ray from `pos` (world cells) in unit step `dir`
/// until a solid cell is hit or [`MAX_PROBE_CELLS`] is exhausted. Returns the
/// distance in cells. Out-of-bounds cells read as `CellKind::Empty`
/// ([`Map::cell_at`]'s own contract), so a probe that exits the authored map
/// keeps marching (treated as open air) rather than stopping at the wall of
/// the level's bounding box.
fn probe_ray(map: &Map, pos: [f32; 3], dir: (i32, i32, i32)) -> f32 {
    let mut x = pos[0].floor() as i32;
    let mut y = pos[1].floor() as i32;
    let mut z = pos[2].floor() as i32;
    let steps = MAX_PROBE_CELLS as i32;
    for step in 1..=steps {
        x += dir.0;
        y += dir.1;
        z += dir.2;
        if map.cell_at(x, y, z).is_solid() {
            return step as f32;
        }
    }
    MAX_PROBE_CELLS
}

/// `true` when the upward probe reached the top of the authored map (or ran
/// out of probe range) without hitting a solid cell — treated as open sky.
/// A tall INDOOR room (a hit far overhead) is not open sky; a room with no
/// roof at all, or one so far up the probe never found it, is.
fn probes_open_sky(map: &Map, pos: [f32; 3], up_distance: f32) -> bool {
    if up_distance >= MAX_PROBE_CELLS {
        return true;
    }
    let (_, _, sz) = map.size();
    pos[2].floor() as i32 + up_distance as i32 >= sz as i32
}

/// Estimate the local room shape at `pos` (world cells) by ray-probing the
/// level grid in the six axis directions. `absorption` is the level's own
/// floor/wall material blend, applied uniformly across the level for now —
/// per-position material sampling (a carpeted alcove inside a stone hall) is
/// future editor work; it does not change the level format, so nothing here
/// blocks it.
pub fn probe_room(map: &Map, pos: [f32; 3], absorption: f32) -> RoomProbe {
    let east = probe_ray(map, pos, (1, 0, 0));
    let west = probe_ray(map, pos, (-1, 0, 0));
    let north = probe_ray(map, pos, (0, 1, 0));
    let south = probe_ray(map, pos, (0, -1, 0));
    let up = probe_ray(map, pos, (0, 0, 1));
    let down = probe_ray(map, pos, (0, 0, -1));
    RoomProbe {
        east,
        west,
        north,
        south,
        up,
        down,
        open_sky: probes_open_sky(map, pos, up),
        absorption,
    }
    .clamped()
}

/// Number of distinct solid-cell "walls" crossed between a source and the
/// listener at which the line is treated as fully occluded. Audio occlusion
/// is a perceptual on/off-ish cue, not a physical transmission-loss model —
/// this saturates fast so a single ordinary wall already reads as solidly
/// "behind something" (`re_audio::spatial::OCCLUSION_GAIN_FLOOR` is the
/// floor a fully-occluded source is still heard at, so it is never actually
/// silent).
const OCCLUSION_FULL_WALLS: f32 = 2.0;

/// How much solid wall sits directly between `from` and `to` (world cells),
/// in `[0, 1]` — sampled along the straight line between them at roughly one
/// sample per cell, counting distinct solid RUNS (not every sample inside a
/// thick wall, which would otherwise make a thicker wall count as MORE
/// occluded than a thin one purely because more samples landed inside it).
/// `0.0` = clear line of sight.
pub fn occlusion_between(map: &Map, from: [f32; 3], to: [f32; 3]) -> f32 {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if dist < 1e-3 {
        return 0.0;
    }
    // Roughly one sample per cell along the segment (never fewer than 2, so
    // even a very short hop still checks its one midpoint).
    let steps = (dist.ceil() as usize).max(2);
    let mut last_solid = false;
    let mut solid_runs = 0u32;
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        let x = (from[0] + dx * t).floor() as i32;
        let y = (from[1] + dy * t).floor() as i32;
        let z = (from[2] + dz * t).floor() as i32;
        let solid = map.cell_at(x, y, z).is_solid();
        if solid && !last_solid {
            solid_runs += 1;
        }
        last_solid = solid;
    }
    (solid_runs as f32 / OCCLUSION_FULL_WALLS).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::{CellKind, Map};

    fn empty_map(w: usize, h: usize, d: usize) -> Map {
        Map::new(w, h, d)
    }

    #[test]
    fn probe_ray_stops_at_the_first_solid_cell() {
        let mut m = empty_map(20, 20, 10);
        m.set(15, 5, 5, CellKind::WallSolid);
        let d = probe_ray(&m, [5.5, 5.5, 5.5], (1, 0, 0));
        // From cell 5 to the wall at cell 15 is 10 steps.
        assert!((d - 10.0).abs() < 1e-4, "{d}");
    }

    #[test]
    fn probe_ray_caps_at_max_probe_cells_in_open_space() {
        let m = empty_map(200, 200, 10);
        let d = probe_ray(&m, [5.5, 5.5, 5.5], (1, 0, 0));
        assert_eq!(d, MAX_PROBE_CELLS);
    }

    #[test]
    fn probe_ray_treats_out_of_bounds_as_open() {
        // A small map: a ray marching past its edge must not stop there.
        let m = empty_map(4, 4, 4);
        let d = probe_ray(&m, [1.5, 1.5, 1.5], (1, 0, 0));
        assert_eq!(
            d, MAX_PROBE_CELLS,
            "an unbounded exterior must not read as a wall"
        );
    }

    #[test]
    fn a_closed_box_probes_as_a_small_closet_not_open_sky() {
        let mut m = empty_map(20, 20, 10);
        for x in 0..20 {
            for y in 0..20 {
                m.set(x, y, 0, CellKind::WallSolid);
                m.set(x, y, 5, CellKind::WallSolid);
            }
        }
        for x in 0..20 {
            for z in 1..5 {
                m.set(x, 0, z, CellKind::WallSolid);
                m.set(x, 5, z, CellKind::WallSolid);
            }
        }
        for y in 0..20 {
            for z in 1..5 {
                m.set(0, y, z, CellKind::WallSolid);
                m.set(5, y, z, CellKind::WallSolid);
            }
        }
        let probe = probe_room(&m, [2.5, 2.5, 2.5], 0.2);
        assert!(!probe.open_sky, "{probe:?}");
        assert!(probe.up < 4.0 && probe.east < 4.0, "{probe:?}");
    }

    #[test]
    fn a_room_with_no_roof_probes_as_open_sky() {
        let mut m = empty_map(20, 20, 10);
        for x in 0..20 {
            for y in 0..20 {
                m.set(x, y, 0, CellKind::WallSolid);
                // No ceiling set at all — up stays open.
            }
        }
        let probe = probe_room(&m, [2.5, 2.5, 1.5], 0.2);
        assert!(probe.open_sky, "{probe:?}");
    }

    #[test]
    fn occlusion_is_zero_with_a_clear_line_of_sight() {
        let m = empty_map(20, 20, 10);
        let o = occlusion_between(&m, [1.5, 1.5, 1.5], [8.5, 1.5, 1.5]);
        assert_eq!(o, 0.0);
    }

    #[test]
    fn one_wall_between_source_and_listener_gives_partial_occlusion() {
        let mut m = empty_map(20, 20, 10);
        for z in 0..10 {
            m.set(5, 1, z, CellKind::WallSolid);
        }
        let o = occlusion_between(&m, [1.5, 1.5, 1.5], [8.5, 1.5, 1.5]);
        assert!(o > 0.0 && o < 1.0, "{o}");
    }

    #[test]
    fn two_walls_saturate_to_full_occlusion() {
        let mut m = empty_map(20, 20, 10);
        for z in 0..10 {
            m.set(4, 1, z, CellKind::WallSolid);
            m.set(6, 1, z, CellKind::WallSolid);
        }
        let o = occlusion_between(&m, [1.5, 1.5, 1.5], [9.5, 1.5, 1.5]);
        assert!((o - 1.0).abs() < 1e-4, "{o}");
    }

    #[test]
    fn a_thick_wall_does_not_read_as_more_occluded_than_a_thin_one() {
        let mut m1 = empty_map(20, 20, 10);
        for z in 0..10 {
            m1.set(5, 1, z, CellKind::WallSolid);
        }
        let thin = occlusion_between(&m1, [1.5, 1.5, 1.5], [8.5, 1.5, 1.5]);

        let mut m2 = empty_map(20, 20, 10);
        for z in 0..10 {
            for x in 4..7 {
                m2.set(x, 1, z, CellKind::WallSolid);
            }
        }
        let thick = occlusion_between(&m2, [1.5, 1.5, 1.5], [8.5, 1.5, 1.5]);
        assert!((thin - thick).abs() < 1e-4, "thin {thin} vs thick {thick}");
    }

    #[test]
    fn zero_distance_is_never_occluded() {
        let m = empty_map(10, 10, 10);
        assert_eq!(occlusion_between(&m, [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]), 0.0);
    }
}
