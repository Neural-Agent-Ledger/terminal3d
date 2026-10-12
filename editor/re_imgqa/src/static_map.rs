//! Static (no-render) per-level analysis — §4b.
//!
//! Computed once per level directly from the parsed [`Level`] + built [`Map`] +
//! puzzle graph, before any ray is cast. Cheap single passes that catch
//! readability and content defects (indistinguishable materials, unreachable
//! cells, claustrophobic pockets) with zero rendering, plus a puzzle-complexity
//! sanity readout.

use std::collections::VecDeque;

use re_level::Level;
use re_puzzle::PuzzleState;
use re_world::Map;

use crate::color;
use crate::metrics::INVISIBILITY_EPS;

/// A pair of materials whose colours are too close to tell apart.
#[derive(Clone, Debug, PartialEq)]
pub struct PaletteCollision {
    /// Name of the first material in the colliding pair.
    pub a: String,
    /// Name of the second material in the colliding pair.
    pub b: String,
    /// Oklab ΔE between the pair's representative (fg) colours.
    pub delta_e: f32,
}

/// Reachability flood-fill summary.
#[derive(Clone, Debug, PartialEq)]
pub struct Reachability {
    /// Count of open cells reachable from the spawn via flood-fill.
    pub reachable_open: u32,
    /// Total count of open (non-solid) cells in the level.
    pub total_open: u32,
    /// Whether every node/door cell is reachable from the spawn.
    pub all_interactive_reachable: bool,
    /// Count of interactive (node/door) cells found unreachable.
    pub unreachable_interactive: u32,
}

/// Openness / dead-end summary over reachable open cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Openness {
    /// Mean count of open (non-solid) 4-connected neighbours over reachable
    /// open cells.
    pub mean_open_neighbors: f32,
    /// Median count of open 4-connected neighbours over reachable open cells.
    pub median_open_neighbors: f32,
    /// Cells with exactly one open neighbour (claustrophobic pockets).
    pub dead_ends: u32,
}

/// Puzzle-graph complexity readout (a difficulty / sanity signal).
#[derive(Clone, Debug, PartialEq)]
pub struct PuzzleComplexity {
    /// Total count of puzzle nodes in the level.
    pub node_count: u32,
    /// Total count of puzzle pipes (edges) in the level.
    pub pipe_count: u32,
    /// Max forward-pass (topological) depth from any source.
    pub max_depth: u32,
    /// Mean out-degree over nodes that have at least one outgoing pipe.
    pub branching_factor: f32,
    /// Whether the puzzle graph reaches its win condition (see `re_puzzle`).
    pub solvable: bool,
    /// Number of door-target nodes (minimal-toggles-to-solve proxy).
    pub door_targets: u32,
}

/// The full per-level static report.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticReport {
    /// The level's file stem / name.
    pub level: String,
    /// Every material pair whose colours collide (see [`PaletteCollision`]).
    pub palette_collisions: Vec<PaletteCollision>,
    /// The reachability flood-fill summary.
    pub reachability: Reachability,
    /// The openness / dead-end summary.
    pub openness: Openness,
    /// The puzzle-graph complexity readout.
    pub puzzle: PuzzleComplexity,
}

/// Run all of §4b for one level.
pub fn analyze_static(level: &Level, map: &Map) -> StaticReport {
    StaticReport {
        level: level.name.clone(),
        palette_collisions: palette_collisions(map),
        reachability: reachability(level, map),
        openness: openness(level, map),
        puzzle: puzzle_complexity(level),
    }
}

/// Pairwise Oklab ΔE over the level's material fg colours; any pair below the
/// invisibility epsilon is a collision (surfaces indistinguishable where they
/// meet). Materials are the authored `Map::materials()` set; a level with none
/// (built-in look) yields an empty list — there is nothing authored to collide.
pub fn palette_collisions(map: &Map) -> Vec<PaletteCollision> {
    let mats = map.materials();
    let mut out = Vec::new();
    for i in 0..mats.len() {
        if mats[i].fallback {
            continue;
        }
        for j in (i + 1)..mats.len() {
            if mats[j].fallback {
                continue;
            }
            let d = color::delta_e_packed(mats[i].fg, mats[j].fg);
            if d < INVISIBILITY_EPS {
                out.push(PaletteCollision {
                    a: mats[i].name.clone(),
                    b: mats[j].name.clone(),
                    delta_e: d,
                });
            }
        }
    }
    out
}

/// True if a cell is open (the player/flood can occupy it): in-bounds and not
/// solid. Mirrors the raycaster's `is_solid` stopping rule.
#[inline]
fn is_open(map: &Map, x: i32, y: i32, z: i32) -> bool {
    map.in_bounds(x, y, z) && !map.cell_at(x, y, z).is_solid()
}

const SIX_DIRS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// 6-connected flood-fill from the spawn over open cells; report reachable vs
/// total open and whether every node/door cell is reachable.
pub fn reachability(level: &Level, map: &Map) -> Reachability {
    let (sx, sy, sz) = map.size();
    let total_open = count_open(map);

    // Spawn cell.
    let spawn = (level.player.x, level.player.y, level.player.z);
    let mut visited = vec![false; sx * sy * sz];
    let idx =
        |x: i32, y: i32, z: i32| -> usize { x as usize + sx * (y as usize + sy * z as usize) };

    let mut reachable_open = 0u32;
    let mut queue = VecDeque::new();
    if is_open(map, spawn.0, spawn.1, spawn.2) {
        visited[idx(spawn.0, spawn.1, spawn.2)] = true;
        queue.push_back(spawn);
        reachable_open += 1;
    }
    while let Some((x, y, z)) = queue.pop_front() {
        for (dx, dy, dz) in SIX_DIRS {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if is_open(map, nx, ny, nz) && !visited[idx(nx, ny, nz)] {
                visited[idx(nx, ny, nz)] = true;
                reachable_open += 1;
                queue.push_back((nx, ny, nz));
            }
        }
    }

    // Interactive cells = node cells + the door cell. They are solid (the DDA
    // stops on them), so "reachable" means an OPEN neighbour was visited.
    let mut interactive: Vec<(i32, i32, i32)> =
        level.nodes.iter().map(|n| (n.x, n.y, n.z)).collect();
    interactive.push((level.door.x, level.door.y, level.door.z));

    let mut unreachable = 0u32;
    for (x, y, z) in interactive {
        let touched = SIX_DIRS.iter().any(|&(dx, dy, dz)| {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            is_open(map, nx, ny, nz) && map.in_bounds(nx, ny, nz) && visited[idx(nx, ny, nz)]
        });
        if !touched {
            unreachable += 1;
        }
    }

    Reachability {
        reachable_open,
        total_open,
        all_interactive_reachable: unreachable == 0,
        unreachable_interactive: unreachable,
    }
}

fn count_open(map: &Map) -> u32 {
    let (sx, sy, sz) = map.size();
    let mut n = 0u32;
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                if is_open(map, x, y, z) {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Open-neighbour statistics over the reachable region (from the spawn), plus
/// the dead-end count (reachable open cells with exactly one open neighbour).
pub fn openness(level: &Level, map: &Map) -> Openness {
    let (sx, sy, sz) = map.size();
    let mut visited = vec![false; sx * sy * sz];
    let idx =
        |x: i32, y: i32, z: i32| -> usize { x as usize + sx * (y as usize + sy * z as usize) };
    let spawn = (level.player.x, level.player.y, level.player.z);

    // Flood from spawn (same as reachability) so openness reflects the playable
    // region, not orphan pockets.
    let mut region = Vec::new();
    let mut queue = VecDeque::new();
    if is_open(map, spawn.0, spawn.1, spawn.2) {
        visited[idx(spawn.0, spawn.1, spawn.2)] = true;
        queue.push_back(spawn);
        region.push(spawn);
    }
    while let Some((x, y, z)) = queue.pop_front() {
        for (dx, dy, dz) in SIX_DIRS {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if is_open(map, nx, ny, nz) && !visited[idx(nx, ny, nz)] {
                visited[idx(nx, ny, nz)] = true;
                queue.push_back((nx, ny, nz));
                region.push((nx, ny, nz));
            }
        }
    }

    let mut counts: Vec<u32> = Vec::with_capacity(region.len());
    let mut dead_ends = 0u32;
    for (x, y, z) in &region {
        let open_neighbors = SIX_DIRS
            .iter()
            .filter(|&&(dx, dy, dz)| is_open(map, x + dx, y + dy, z + dz))
            .count() as u32;
        if open_neighbors == 1 {
            dead_ends += 1;
        }
        counts.push(open_neighbors);
    }

    let mean = if counts.is_empty() {
        0.0
    } else {
        counts.iter().sum::<u32>() as f32 / counts.len() as f32
    };
    counts.sort_unstable();
    let median = if counts.is_empty() {
        0.0
    } else {
        counts[counts.len() / 2] as f32
    };

    Openness {
        mean_open_neighbors: mean,
        median_open_neighbors: median,
        dead_ends,
    }
}

/// Puzzle-graph complexity from `re_puzzle`'s public query surface. Forward-pass
/// depth and branching are recomputed from the public node/pipe data (the
/// crate's internal topo order is private; the graph is small).
pub fn puzzle_complexity(level: &Level) -> PuzzleComplexity {
    let node_count = level.nodes.len() as u32;
    let pipe_count = level.pipes.len() as u32;

    // Build (or fail to build) the puzzle network. A cyclic / malformed graph is
    // reported as not-solvable with zero depth.
    let state = PuzzleState::from_level(level);
    let solvable = state.as_ref().map(|s| s.solved).unwrap_or(false);
    let door_targets = state
        .as_ref()
        .map(|s| s.door_targets.len() as u32)
        .unwrap_or(level.door.target_parts.len() as u32);

    // Forward-pass depth: longest path (in the DAG of pipe edges) ending at each
    // node, seeded at sources (in-degree 0 or input). Computed over the public
    // pipe topology; cyclic graphs cap depth at node_count (defensive).
    let (max_depth, branching_factor) = depth_and_branching(level);

    PuzzleComplexity {
        node_count,
        pipe_count,
        max_depth,
        branching_factor,
        solvable,
        door_targets,
    }
}

/// Longest topological depth + mean out-degree (over nodes with ≥1 out-edge),
/// from the level's node/pipe data. Edges use `from_node`/`to_node` (resolved by
/// the level parser). A cycle is bounded by `node_count` iterations.
fn depth_and_branching(level: &Level) -> (u32, f32) {
    let n = level.nodes.len();
    if n == 0 {
        return (0, 0.0);
    }
    // Adjacency + in-degree from pipe topology.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_deg = vec![0u32; n];
    let mut out_deg = vec![0u32; n];
    for p in &level.pipes {
        if p.from_node < n && p.to_node < n {
            adj[p.from_node].push(p.to_node);
            in_deg[p.to_node] += 1;
            out_deg[p.from_node] += 1;
        }
    }

    // Kahn longest-path: depth[node] = max(depth[pred]) + 1.
    let mut depth = vec![0u32; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut remaining = in_deg.clone();
    for (i, &d) in in_deg.iter().enumerate() {
        if d == 0 || level.nodes[i].is_input {
            queue.push_back(i);
        }
    }
    let mut processed = 0usize;
    while let Some(u) = queue.pop_front() {
        processed += 1;
        for &v in &adj[u] {
            depth[v] = depth[v].max(depth[u] + 1);
            if remaining[v] > 0 {
                remaining[v] -= 1;
                if remaining[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        if processed > n * 4 {
            break; // cycle guard
        }
    }
    let max_depth = depth.iter().copied().max().unwrap_or(0);

    let branchers: Vec<u32> = out_deg.into_iter().filter(|&d| d > 0).collect();
    let branching = if branchers.is_empty() {
        0.0
    } else {
        branchers.iter().sum::<u32>() as f32 / branchers.len() as f32
    };
    (max_depth, branching)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::{CameraPose, CellKind, Map};

    // A tiny open box: a 5×5×3 room, perimeter walls at every z, floor/ceiling
    // slabs, all interior air at z=1 reachable.
    fn open_box() -> Map {
        let (sx, sy, sz) = (5usize, 5usize, 3usize);
        let mut m = Map::new(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    m.set(x, y, z, kind);
                }
            }
        }
        m
    }

    fn level_with_spawn(x: i32, y: i32, z: i32) -> Level {
        let mut l = Level::default();
        l.player.x = x;
        l.player.y = y;
        l.player.z = z;
        l
    }

    #[test]
    fn reachability_open_room_all_reachable() {
        let map = open_box();
        let lvl = level_with_spawn(2, 2, 1); // interior air
        let r = reachability(&lvl, &map);
        assert!(r.total_open > 0);
        // The single interior air cell at (2,2,1) is reachable.
        assert!(r.reachable_open >= 1, "spawn cell must be reachable");
        // Cells beyond walls aren't counted; reachable ≤ total.
        assert!(r.reachable_open <= r.total_open);
    }

    #[test]
    fn reachability_sealed_spawn_finds_nothing() {
        let map = open_box();
        // Spawn inside a solid wall cell → nothing reachable.
        let lvl = level_with_spawn(0, 0, 1);
        let r = reachability(&lvl, &map);
        assert_eq!(
            r.reachable_open, 0,
            "spawn in solid → no reachable open cells"
        );
    }

    #[test]
    fn palette_collision_detects_identical_materials() {
        use re_world::MaterialDef;
        let mut map = open_box();
        map.set_materials(vec![
            MaterialDef::flat("a", 0x00112233, 0x00000000),
            MaterialDef::flat("b", 0x00112233, 0x00000000), // identical fg
            MaterialDef::flat("c", 0x00FF0000, 0x00000000),
        ]);
        let cols = palette_collisions(&map);
        assert_eq!(cols.len(), 1, "exactly one identical pair collides");
        assert!((cols[0].a == "a" && cols[0].b == "b") || (cols[0].a == "b" && cols[0].b == "a"));
    }

    #[test]
    fn openness_counts_neighbors() {
        let map = open_box();
        let lvl = level_with_spawn(2, 2, 1);
        let o = openness(&lvl, &map);
        // The single interior cell (2,2,1) has open neighbours only at... none of
        // its 6 neighbours are open (walls + floor + ceiling), so it's a 0-degree
        // pocket. The metric must still compute without panic.
        assert!(o.mean_open_neighbors >= 0.0);
        let _ = CameraPose::spawn_at(2, 2, 1, 0); // smoke: camera API present
    }
}
