//! Camera-orbit pose generation — §5.
//!
//! From a level's [`Map`] + spawn, enumerate passable spawn cells, decimate to a
//! sample budget with a fixed stride (deterministic — no RNG), then for each kept
//! position emit a fan of `yaws × pitches` poses. The level's own spawn pose is
//! always included and tagged `spawn`.

use re_world::{CameraPose, Map};

/// One generated pose plus its identity for reporting / photo filenames.
#[derive(Clone, Debug)]
pub struct OrbitPose {
    /// Stable per-level pose index (assignment order). Drives deterministic
    /// result collection and the photo filename.
    pub index: usize,
    /// A short id like `spawn`, or `p3_y08_pup` (position 3, yaw 8, pitch up).
    pub id: String,
    /// The camera pose to raycast.
    pub pose: CameraPose,
    /// True for the level's authored spawn pose.
    pub is_spawn: bool,
}

/// Pitch offset selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PitchMode {
    /// Level pitch only.
    LevelOnly,
    /// Level / up / down (3 pitches).
    Three,
}

/// Orbit generation parameters (from the CLI).
#[derive(Clone, Copy, Debug)]
pub struct OrbitParams {
    /// Position sample budget per level.
    pub positions: usize,
    /// Yaw directions per position.
    pub yaws: usize,
    /// Which pitch offsets to sample at each kept position.
    pub pitch: PitchMode,
}

impl Default for OrbitParams {
    fn default() -> Self {
        OrbitParams {
            positions: 40,
            yaws: 16,
            pitch: PitchMode::Three,
        }
    }
}

/// The up/down pitch magnitude (radians) for the non-level pitch offsets.
pub const PITCH_OFFSET: f32 = std::f32::consts::FRAC_PI_4 * 0.6; // ~27°

/// Enumerate passable spawn cells of `map` at the spawn Z: a cell is passable if
/// it is non-solid AND the cell directly above it is non-solid (head room). The
/// scan is row-major (z fixed = spawn_z, then y, then x) so the order is stable.
pub fn passable_cells(map: &Map, spawn_z: i32) -> Vec<(i32, i32, i32)> {
    let (sx, sy, _sz) = map.size();
    let mut out = Vec::new();
    for y in 0..sy as i32 {
        for x in 0..sx as i32 {
            let here = map.cell_at(x, y, spawn_z);
            let above = map.cell_at(x, y, spawn_z + 1);
            if !here.is_solid() && !above.is_solid() && map.in_bounds(x, y, spawn_z) {
                out.push((x, y, spawn_z));
            }
        }
    }
    out
}

/// Decimate a list to at most `budget` entries via an even fixed stride. The
/// first element is always kept; the stride is `ceil(len / budget)`. Deterministic.
pub fn decimate<T: Clone>(items: &[T], budget: usize) -> Vec<T> {
    if budget == 0 || items.is_empty() {
        return Vec::new();
    }
    if items.len() <= budget {
        return items.to_vec();
    }
    let stride = items.len().div_ceil(budget);
    items.iter().step_by(stride).take(budget).cloned().collect()
}

/// Build the full pose list for a level. The authored `spawn` pose is index 0;
/// the decimated passable positions each contribute `yaws × pitches` poses.
pub fn generate(map: &Map, spawn: CameraPose, params: &OrbitParams) -> Vec<OrbitPose> {
    let mut poses = Vec::new();
    let mut index = 0usize;

    // 0. The authored spawn pose, always first and tagged.
    poses.push(OrbitPose {
        index,
        id: "spawn".to_string(),
        pose: spawn,
        is_spawn: true,
    });
    index += 1;

    let spawn_z = spawn.pos.z.floor() as i32;
    let cells = passable_cells(map, spawn_z);
    let kept = decimate(&cells, params.positions);

    let pitches: &[(f32, &str)] = match params.pitch {
        PitchMode::LevelOnly => &[(0.0, "lvl")],
        PitchMode::Three => &[(0.0, "lvl"), (PITCH_OFFSET, "pup"), (-PITCH_OFFSET, "pdn")],
    };

    let yaw_count = params.yaws.max(1);
    for (pi, &(cx, cy, cz)) in kept.iter().enumerate() {
        for yi in 0..yaw_count {
            let yaw = (yi as f32 / yaw_count as f32) * std::f32::consts::TAU;
            for &(pitch, ptag) in pitches {
                let pose = CameraPose {
                    pos: glam::Vec3::new(cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5),
                    yaw,
                    pitch,
                };
                poses.push(OrbitPose {
                    index,
                    id: format!("p{pi}_y{yi:02}_{ptag}"),
                    pose,
                    is_spawn: false,
                });
                index += 1;
            }
        }
    }
    poses
}

/// Group pose indices that share a position and pitch, in ascending yaw order,
/// for the shimmer pass. Returns lists of `(yaw_step, pose_indices)`. The spawn
/// pose is excluded (it has no yaw siblings).
pub fn shimmer_groups(poses: &[OrbitPose], params: &OrbitParams) -> Vec<(f32, Vec<usize>)> {
    let yaw_count = params.yaws.max(1);
    let yaw_step = std::f32::consts::TAU / yaw_count as f32;
    let pitches = match params.pitch {
        PitchMode::LevelOnly => 1usize,
        PitchMode::Three => 3,
    };
    let mut groups = Vec::new();
    // Non-spawn poses are laid out as [position][yaw][pitch]. Walk that layout.
    let non_spawn: Vec<&OrbitPose> = poses.iter().filter(|p| !p.is_spawn).collect();
    let per_position = yaw_count * pitches;
    if per_position == 0 {
        return groups;
    }
    let n_positions = non_spawn.len() / per_position;
    for pos in 0..n_positions {
        for pitch in 0..pitches {
            // Collect this (position, pitch) across all yaws, in yaw order.
            let mut idxs = Vec::with_capacity(yaw_count);
            for yaw in 0..yaw_count {
                let local = pos * per_position + yaw * pitches + pitch;
                idxs.push(non_spawn[local].index);
            }
            groups.push((yaw_step, idxs));
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::CellKind;

    // A 3×3×3 map with exactly one open interior column at (1,1): air at z=1 with
    // air above at z=2 → exactly one passable cell at spawn_z=1.
    fn one_open_cell_map() -> Map {
        let mut m = Map::new(3, 3, 3);
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    // Make everything solid except the (1,1,1) and (1,1,2) column.
                    let open = x == 1 && y == 1 && (z == 1 || z == 2);
                    if !open {
                        m.set(x, y, z, CellKind::WallSolid);
                    }
                }
            }
        }
        m
    }

    #[test]
    fn one_open_cell_yields_expected_pose_count() {
        let map = one_open_cell_map();
        let spawn = CameraPose::spawn_at(1, 1, 1, 0);
        let params = OrbitParams {
            positions: 40,
            yaws: 16,
            pitch: PitchMode::Three,
        };
        let poses = generate(&map, spawn, &params);

        let cells = passable_cells(&map, 1);
        assert_eq!(cells.len(), 1, "exactly one passable cell expected");

        // 1 spawn + 1 position × 16 yaws × 3 pitches = 1 + 48 = 49.
        assert_eq!(poses.len(), 1 + 16 * 3);
        // Spawn present and tagged.
        assert!(poses[0].is_spawn);
        assert_eq!(poses[0].id, "spawn");
        assert_eq!(poses.iter().filter(|p| p.is_spawn).count(), 1);
        // Indices are dense and ordered.
        for (i, p) in poses.iter().enumerate() {
            assert_eq!(p.index, i);
        }
    }

    #[test]
    fn no_pitch_yields_one_pitch() {
        let map = one_open_cell_map();
        let spawn = CameraPose::spawn_at(1, 1, 1, 0);
        let params = OrbitParams {
            positions: 40,
            yaws: 16,
            pitch: PitchMode::LevelOnly,
        };
        let poses = generate(&map, spawn, &params);
        assert_eq!(poses.len(), 1 + 16, "level-only pitch → 16 poses + spawn");
    }

    #[test]
    fn decimation_respects_budget() {
        let items: Vec<i32> = (0..1000).collect();
        let kept = decimate(&items, 40);
        assert!(
            kept.len() <= 40,
            "must not exceed budget, got {}",
            kept.len()
        );
        assert_eq!(kept[0], 0, "first element always kept");
        // Even stride: deterministic.
        let kept2 = decimate(&items, 40);
        assert_eq!(kept, kept2, "decimation must be deterministic");
    }

    #[test]
    fn decimation_keeps_small_lists_intact() {
        let items: Vec<i32> = (0..5).collect();
        assert_eq!(decimate(&items, 40), items);
    }

    #[test]
    fn shimmer_groups_cover_yaw_sequences() {
        let map = one_open_cell_map();
        let spawn = CameraPose::spawn_at(1, 1, 1, 0);
        let params = OrbitParams {
            positions: 40,
            yaws: 16,
            pitch: PitchMode::Three,
        };
        let poses = generate(&map, spawn, &params);
        let groups = shimmer_groups(&poses, &params);
        // 1 position × 3 pitches = 3 groups, each of 16 yaw-ordered poses.
        assert_eq!(groups.len(), 3);
        for (_step, idxs) in &groups {
            assert_eq!(idxs.len(), 16);
        }
    }
}
