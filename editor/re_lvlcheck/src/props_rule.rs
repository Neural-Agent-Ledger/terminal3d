//! Check 13 — `props`: the set-dressing prop kit must never touch the puzzle
//! (set-dressing §3.7).
//!
//! 1. **Expansion findings.** Everything the generator reported while expanding
//!    (ring keep-out hits, stair headroom, gallery jump reach, holograms in a
//!    switch column → FAIL; skipped cells, clamps → WARN). A HARD keep-out hit
//!    never gets here: it is a load error.
//! 2. **Connectivity.** Flood the walkable graph from the spawn on the room
//!    without props and with them. Every switch stand-cell and the door that the
//!    bare room reaches must still be reached (FAIL), and the props may remove
//!    only their own footprint (a cut-off pocket of more than 2 cells is a WARN).
//! 3. **No new shortcuts.** With every prop EXCEPT the declared stairs, no
//!    authored (non-prop) surface may become reachable that the bare room could
//!    not reach — debris and galleries must not become ladders (FAIL).
//! 4. **Budgets.** Lights ≤ 4 (FAIL), emitter particles < 4096 (FAIL; WARN from
//!    3000), trace line segments ≤ 400 (WARN).
//! 5. **Colour law.** A powered prop lights in a hue some puzzle element carries
//!    (WARN otherwise).
//!
//! The walk model is deliberately the linter's, not the engine's: a stand cell is
//! a non-blocking cell with support below (or a ramp cell) and a non-blocking cell
//! above; moves are 4-neighbour at the same height, one cell up (a step or a
//! jump), and any fall. It is the same model for both floods, so a difference
//! between them is the props' doing.

use std::collections::{HashSet, VecDeque};

use re_core::Shape;
use re_level::props::{self, IssueLevel, PropExpansion};
use re_level::Level;
use re_world::{CellKind, Map};

use crate::{Check, Location, Severity, MAX_POINT_LIGHTS};

/// Particle ceiling for the level's emitters (§3.7).
pub const MAX_PARTICLES: u32 = 4096;
/// Particle count that earns a WARN.
pub const WARN_PARTICLES: u32 = 3000;
/// Trace line segment budget (WARN above).
pub const MAX_SEGMENTS: usize = 400;

type C3 = (i32, i32, i32);

fn blocking(m: &Map, c: C3) -> bool {
    let k = m.cell_at(c.0, c.1, c.2);
    (k.is_solid() || matches!(k, CellKind::Floor | CellKind::Ceiling)) && !is_ramp(m, c)
}

fn is_ramp(m: &Map, c: C3) -> bool {
    m.cell_at(c.0, c.1, c.2).is_solid() && matches!(m.shape_at(c.0, c.1, c.2), Shape::RampZ(_))
}

fn solid_like(m: &Map, c: C3) -> bool {
    let k = m.cell_at(c.0, c.1, c.2);
    k.is_solid() || matches!(k, CellKind::Floor | CellKind::Ceiling)
}

/// Can the player stand with their feet in `c`?
fn stand(m: &Map, c: C3) -> bool {
    m.in_bounds(c.0, c.1, c.2)
        && !blocking(m, c)
        && (is_ramp(m, c) || solid_like(m, (c.0, c.1, c.2 - 1)))
        && !blocking(m, (c.0, c.1, c.2 + 1))
}

/// Where a player entering the air column at `c` comes to rest.
fn land(m: &Map, c: C3) -> Option<C3> {
    let mut z = c.2;
    while z >= 0 {
        let p = (c.0, c.1, z);
        if !m.in_bounds(p.0, p.1, p.2) || blocking(m, p) {
            return None;
        }
        if stand(m, p) {
            return Some(p);
        }
        z -= 1;
    }
    None
}

/// Every stand cell reachable from the spawn.
fn flood(m: &Map, spawn: C3) -> HashSet<C3> {
    let mut seen = HashSet::new();
    let Some(start) = land(m, spawn).or_else(|| land(m, (spawn.0, spawn.1, spawn.2 + 1))) else {
        return seen;
    };
    let mut q = VecDeque::from([start]);
    seen.insert(start);
    while let Some(c) = q.pop_front() {
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (c.0 + dx, c.1 + dy);
            let mut next = Vec::new();
            // Same height, or a fall off the edge.
            if !blocking(m, (nx, ny, c.2)) && !blocking(m, (nx, ny, c.2 + 1)) {
                if let Some(l) = land(m, (nx, ny, c.2)) {
                    next.push(l);
                }
            }
            // One cell up (a step, a ramp, a jump) — needs headroom above the start.
            let up = (nx, ny, c.2 + 1);
            if !blocking(m, (c.0, c.1, c.2 + 2)) && stand(m, up) {
                next.push(up);
            }
            for n in next {
                if seen.insert(n) {
                    q.push_back(n);
                }
            }
        }
    }
    seen
}

/// Stand cells from which switch `s` can be worked (own column + 4 neighbours,
/// reach band as `switch_physically_reachable`).
fn switch_stands(m: &Map, s: C3, jump: f32) -> Vec<C3> {
    let (_, _, sz) = m.size();
    let mut out = Vec::new();
    for (cx, cy) in [
        (s.0, s.1),
        (s.0 + 1, s.1),
        (s.0 - 1, s.1),
        (s.0, s.1 + 1),
        (s.0, s.1 - 1),
    ] {
        for z in 0..sz as i32 {
            let c = (cx, cy, z);
            let lo = z as f32 - 0.5;
            let hi = z as f32 + jump + 1.5;
            if (lo..=hi).contains(&(s.2 as f32)) && stand(m, c) {
                out.push(c);
            }
        }
    }
    out
}

/// Stand cells next to the door anchor.
fn door_stands(m: &Map, d: C3) -> Vec<C3> {
    let mut out = Vec::new();
    for (cx, cy) in [
        (d.0, d.1),
        (d.0 + 1, d.1),
        (d.0 - 1, d.1),
        (d.0, d.1 + 1),
        (d.0, d.1 - 1),
    ] {
        for z in d.2 - 1..=d.2 + 1 {
            if stand(m, (cx, cy, z)) {
                out.push((cx, cy, z));
            }
        }
    }
    out
}

/// The room with the props stamped, every R1 lit slot filled (the worst case
/// for connectivity), optionally leaving out the declared stairs.
fn with_props(level: &Level, base: &Map, exp: &PropExpansion, stairs: bool) -> Map {
    let mut m = base.clone();
    let mut e = exp.clone();
    if !stairs {
        let st: HashSet<C3> = exp.stair_cells.iter().copied().collect();
        e.cells.retain(|c| !st.contains(&c.pos));
    }
    props::stamp_expansion(level, &e, &mut m);
    for p in &exp.parts {
        let (x, y, z) = p.pos;
        if x >= 0
            && y >= 0
            && z >= 0
            && m.in_bounds(x, y, z)
            && m.cell_at(x, y, z) == CellKind::Empty
        {
            m.set_cell(x as usize, y as usize, z as usize, CellKind::WallSolid, 1);
        }
    }
    m
}

/// Check 13 — see the module docs.
pub(crate) fn check_props(level: &Level, jump: f32) -> Check {
    if level.props.is_empty() {
        return Check::new("props", Severity::Pass, "no [prop] sections");
    }
    let base = level.build_map_base();
    let (exp, errs) = props::expand_level(level, &base);
    let mut fails: Vec<(String, Option<Location>)> = Vec::new();
    let mut warns: Vec<(String, Option<Location>)> = Vec::new();
    for e in errs {
        fails.push((e.msg, Some(Location::SourceLine(e.line))));
    }
    for i in &exp.issues {
        let loc = i.pos.map(Location::cell);
        let msg = format!("[prop {}] {}", i.prop, i.msg);
        match i.level {
            IssueLevel::Fail => fails.push((msg, loc)),
            IssueLevel::Warn => warns.push((msg, loc)),
        }
    }

    // ── connectivity ─────────────────────────────────────────────────────────
    let spawn = (level.player.x, level.player.y, level.player.z);
    let with = with_props(level, &base, &exp, true);
    let r0 = flood(&base, spawn);
    let r1 = flood(&with, spawn);
    if r0.is_empty() {
        // Every connectivity and shortcut rule below compares two floods from the
        // spawn. With no stand cell at the spawn both floods are empty and each rule
        // passes vacuously ("reach 0 -> 0"), a green check that proved nothing.
        fails.push((
            format!(
                "the walk model finds no stand cell at the spawn {spawn:?} (no floor under it, or no \
                 headroom above it): reach is 0, so the prop connectivity rules cannot be checked"
            ),
            Some(Location::cell(spawn)),
        ));
    }
    for (i, p) in level.pipes.iter().enumerate() {
        let Some(s) = p.switch_cell.and_then(|k| p.path.get(k)).copied() else {
            continue;
        };
        let before = switch_stands(&base, s, jump).iter().any(|c| r0.contains(c));
        let after = switch_stands(&with, s, jump).iter().any(|c| r1.contains(c));
        if before && !after {
            fails.push((
                format!(
                    "props cut off the switch of pipe {i} ({}->{}) at {s:?}",
                    p.from, p.to
                ),
                Some(Location::cell(s)),
            ));
        }
    }
    let d = (level.door.x, level.door.y, level.door.z);
    if door_stands(&base, d).iter().any(|c| r0.contains(c))
        && !door_stands(&with, d).iter().any(|c| r1.contains(c))
    {
        fails.push((
            format!("props cut off the door at {d:?}"),
            Some(Location::cell(d)),
        ));
    }
    let footprint: HashSet<C3> = exp
        .cells
        .iter()
        .map(|c| c.pos)
        .chain(exp.parts.iter().map(|p| p.pos))
        .collect();
    let lost: Vec<C3> = {
        let mut v: Vec<C3> = r0
            .iter()
            .filter(|c| {
                !r1.contains(c)
                    && !footprint.contains(c)
                    && !footprint.contains(&(c.0, c.1, c.2 + 1))
            })
            .copied()
            .collect();
        v.sort_unstable();
        v
    };
    if lost.len() > 2 {
        warns.push((
            format!(
                "props cut off {} walkable cell(s) beyond their own footprint (first {:?})",
                lost.len(),
                lost[0]
            ),
            Some(Location::cell(lost[0])),
        ));
    }

    // ── shortcuts: everything but the declared stairs ─────────────────────────
    let no_stairs = if exp.stair_cells.is_empty() {
        with.clone()
    } else {
        with_props(level, &base, &exp, false)
    };
    let r2 = flood(&no_stairs, spawn);
    let prop_cells: HashSet<C3> = footprint;
    let mut shortcut: Vec<C3> = r2
        .iter()
        .filter(|c| !r0.contains(c))
        // Standing ON a prop is the prop's own surface; a shortcut is AUTHORED
        // ground that only became reachable because a prop was in the way.
        .filter(|c| !prop_cells.contains(&(c.0, c.1, c.2 - 1)) && !prop_cells.contains(c))
        .copied()
        .collect();
    shortcut.sort_unstable();
    if let Some(&c) = shortcut.first() {
        fails.push((
            format!(
                "props open {} new authored stand cell(s) the bare room cannot reach — a ladder, not a stair (first {c:?})",
                shortcut.len()
            ),
            Some(Location::cell(c)),
        ));
    }

    // ── budgets ──────────────────────────────────────────────────────────────
    let lights = level.lights.len() + exp.lights.len();
    if lights > MAX_POINT_LIGHTS {
        fails.push((
            format!(
                "{lights} lights with the props' lamps; the renderer honours {MAX_POINT_LIGHTS}"
            ),
            None,
        ));
    }
    let particles: u32 = level.emitters.iter().map(|e| e.max_particles).sum();
    if particles >= MAX_PARTICLES {
        fails.push((
            format!("{particles} emitter particles (limit {MAX_PARTICLES})"),
            None,
        ));
    } else if particles >= WARN_PARTICLES {
        warns.push((
            format!("{particles} emitter particles (warn at {WARN_PARTICLES})"),
            None,
        ));
    }
    if exp.segments > MAX_SEGMENTS {
        warns.push((
            format!(
                "{} prop line segments (budget {MAX_SEGMENTS})",
                exp.segments
            ),
            None,
        ));
    }

    // ── colour law ───────────────────────────────────────────────────────────
    let hues: HashSet<u32> = level
        .nodes
        .iter()
        .map(|n| n.color_rgb & 0xFF_FFFF)
        .chain(level.pipes.iter().map(|p| p.color_rgb & 0xFF_FFFF))
        .chain([0xFF_B040, 0xFF_3030])
        .collect();
    for g in &level.prop_power {
        if !hues.contains(&(g.lit_rgb & 0xFF_FFFF)) {
            warns.push((
                format!(
                    "[prop {}] lights in #{:06X}, a hue no puzzle element carries",
                    g.prop, g.lit_rgb
                ),
                None,
            ));
        }
    }

    let n = level.props.len();
    let summary = format!(
        "{n} prop(s): {} cells, {} lines ({} segments), {} holo(s), {} emitter(s), {} lamp(s); reach {} -> {} stand cells",
        exp.cells.len(),
        exp.lines.len(),
        exp.segments,
        exp.holos.len(),
        exp.emitters.len(),
        exp.lights.len(),
        r0.len(),
        r1.len()
    );
    let (sev, list) = if !fails.is_empty() {
        (Severity::Fail, fails)
    } else if !warns.is_empty() {
        (Severity::Warn, warns)
    } else {
        return Check::new("props", Severity::Pass, summary);
    };
    let detail = format!(
        "{summary} | {}",
        list.iter()
            .map(|(m, _)| m.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    );
    Check::new("props", sev, detail).at_all(list.into_iter().filter_map(|(_, l)| l).collect())
}
