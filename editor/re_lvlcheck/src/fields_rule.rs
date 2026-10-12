//! Rule `fields` — authored `[field]` sound fields.
//!
//! Referential integrity (`bind`/`drive`/`follow` naming a real node, pipe or
//! note) is already a HARD parse error in `re_level::fields::resolve` — a level
//! that reaches this rule already has every field's names resolved, which is a
//! *stronger* guarantee than a lint-time WARN would be (a bad reference cannot
//! ship at all, not even quietly). This rule therefore covers exactly the
//! remaining concerns that are genuinely LINT-level rather than load errors: an
//! authored knob out of its documented range (defence in depth —
//! `re_level` already clamps/rejects these too, so this can only ever
//! corroborate, never catch something `re_level` missed, but it keeps the
//! `fields` report self-contained the way every other rule here is), density
//! overlap at a walkable cell, `tail_f` too close to the solve path, too many
//! Hold-using fields, and a campaign door with no `exit_detune`.

use re_level::{Falloff, FieldRadius, FieldShape, Level, ResolvedField};

use crate::{Check, Location, Severity};

/// How many non-room fields exceeding `OVERLAP_WEIGHT_THRESHOLD` at one cell is
/// still fine ("more than 3 … overlap").
const OVERLAP_MAX: usize = 3;
const OVERLAP_WEIGHT_THRESHOLD: f32 = 0.5;
/// `tail_f` within this many cells of a solve-path element is a WARN.
const TAIL_F_MIN_DISTANCE: f32 = 3.0;
/// More than this many Hold-using fields (`mnemos_hold`, `tail_f`) in one level
/// is a WARN.
const HOLD_FIELD_MAX: usize = 2;

/// A cheap, deliberately approximate heuristic for "is this a campaign level":
/// the shipped corpus names each campaign entry `L<digits>-...`
/// (`L1-first-pulse` … `L12-closing`). `re_lvlcheck` cannot depend on a host
/// game crate (which would own the real level-to-campaign-slot mapping), so
/// this stays a naming-convention proxy — good enough for a WARN, never a FAIL.
fn looks_like_a_campaign_level(level: &Level) -> bool {
    let name = level.name.trim_start_matches('L');
    level.name.starts_with('L') && name.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// A representative world point for a field's shape, for the proximity/overlap
/// heuristics below. `Room` has none (it has no position by design).
fn representative_point(shape: &FieldShape) -> Option<[f32; 3]> {
    match shape {
        FieldShape::Sphere { at } => Some(*at),
        FieldShape::BoxShape { min, max } => Some([
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ]),
        FieldShape::Capsule { points } => points
            .get(points.len() / 2)
            .copied()
            .or_else(|| points.first().copied()),
        FieldShape::Room => None,
    }
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Distance from `p` to a field's shape (0 inside a box). Mirrors the shape
/// vocabulary's "distance used" column closely enough for a static lint; the
/// host game's runtime falloff is the gameplay-authoritative version.
fn shape_distance(shape: &FieldShape, p: [f32; 3]) -> f32 {
    match shape {
        FieldShape::Sphere { at } => dist(*at, p),
        FieldShape::BoxShape { min, max } => {
            let dx = (min[0] - p[0]).max(0.0).max(p[0] - max[0]);
            let dy = (min[1] - p[1]).max(0.0).max(p[1] - max[1]);
            let dz = (min[2] - p[2]).max(0.0).max(p[2] - max[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
        FieldShape::Capsule { points } => points
            .iter()
            .map(|&q| dist(q, p))
            .fold(f32::INFINITY, f32::min),
        FieldShape::Room => 0.0,
    }
}

fn falloff_weight(falloff: Falloff, d: f32, inner: f32, radius: f32) -> f32 {
    if d <= inner {
        return 1.0;
    }
    if radius <= inner {
        return 0.0;
    }
    let t = ((d - inner) / (radius - inner)).clamp(0.0, 1.0);
    match falloff {
        Falloff::Smooth => {
            let s = t * t * (3.0 - 2.0 * t);
            1.0 - s
        }
        Falloff::Linear => 1.0 - t,
        Falloff::Inverse => 1.0 / (1.0 + d * d / radius.max(1e-3).powi(2)),
        Falloff::Hard => {
            if d <= radius {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// A field's static weight at `p`, ignoring puzzle-state gating/drive/phase
/// (this is a WORST-CASE geometry-only estimate — the overlap-density rule
/// wants "could these ever be loud together", not "are they right now").
fn static_weight_at(f: &ResolvedField, p: [f32; 3]) -> f32 {
    if matches!(f.shape, FieldShape::Room) {
        return 0.0; // room fields are handled separately (§4.5 excludes them: "non-room fields").
    }
    let radius = match f.radius {
        FieldRadius::Value(r) => r,
        FieldRadius::Room => return 1.0,
    };
    let d = shape_distance(&f.shape, p);
    falloff_weight(f.falloff, d, f.inner, radius)
}

/// The `fields` rule: lints authored `[field]` sound fields (see the module docs).
#[must_use]
pub fn check_fields(level: &Level) -> Check {
    let fields = level.fields();
    if fields.is_empty() {
        return Check::new("fields", Severity::Pass, "no [field] sections authored");
    }

    let mut warnings: Vec<String> = Vec::new();
    let mut locations: Vec<Location> = Vec::new();

    // -- Hold-count (F2 mnemos_hold, F12 tail_f) ---------------------------
    let hold_count = fields
        .iter()
        .filter(|f| f.archetype == "mnemos_hold" || f.archetype == "tail_f")
        .count();
    if hold_count > HOLD_FIELD_MAX {
        warnings.push(format!(
            "{hold_count} Hold-using fields (mnemos_hold/tail_f) authored, more than the \
             comfort ceiling of {HOLD_FIELD_MAX} (§3.6 rule 4's continuity limits get harder \
             to guarantee the more of them can stack)"
        ));
    }

    // -- tail_f proximity to the solve path ---------------------------------
    let mut solve_path_points: Vec<[f32; 3]> = Vec::new();
    for n in &level.nodes {
        solve_path_points.push([n.x as f32, n.y as f32, n.z as f32]);
    }
    for p in &level.pipes {
        if let Some(idx) = p.switch_cell {
            if let Some(&(x, y, z)) = p.path.get(idx) {
                solve_path_points.push([x as f32, y as f32, z as f32]);
            }
        }
    }
    solve_path_points.push([
        level.door.x as f32,
        level.door.y as f32,
        level.door.z as f32,
    ]);

    for f in fields.iter().filter(|f| f.archetype == "tail_f") {
        let Some(p) = representative_point(&f.shape) else {
            continue;
        };
        if let Some(nearest) = solve_path_points
            .iter()
            .map(|&q| dist(p, q))
            .fold(None, |acc, d| Some(acc.map_or(d, |a: f32| a.min(d))))
        {
            if nearest < TAIL_F_MIN_DISTANCE {
                warnings.push(format!(
                    "[field {}] tail_f sits {nearest:.1} cells from a switch/node/door \
                     (< {TAIL_F_MIN_DISTANCE}) — optional lore risks reading as a solve hint",
                    f.name
                ));
                locations.push(Location::cell((
                    p[0].floor() as i32,
                    p[1].floor() as i32,
                    p[2].floor() as i32,
                )));
            }
        }
    }

    // -- campaign door without an exit_detune -------------------------------
    if looks_like_a_campaign_level(level) && !fields.iter().any(|f| f.archetype == "exit_detune") {
        warnings.push(
            "campaign-named level authors no exit_detune field — the grammar anchor \
             (\"in tune = open\") is silent at every door"
                .to_string(),
        );
    }

    // -- authored-knob sanity (defence in depth; re_level already enforces
    //    these at load) -----------------------------------------------------
    for f in fields {
        if let FieldRadius::Value(r) = f.radius {
            if !(r > 0.0 && r.is_finite()) {
                warnings.push(format!("[field {}] radius {r} is out of range", f.name));
            }
            if f.inner > r {
                warnings.push(format!(
                    "[field {}] inner ({}) exceeds radius ({r})",
                    f.name, f.inner
                ));
            }
        }
    }

    // -- density overlap at walkable cells -----------------------------------
    // Sample every non-solid cell of the built map at roughly head height
    // (matches the archetype table's own listener convention, §2.0's
    // occlusion probe uses the same map). Skip Room-shaped fields (§4.5: "3
    // non-room fields").
    let non_room: Vec<&ResolvedField> = fields
        .iter()
        .filter(|f| !matches!(f.shape, FieldShape::Room))
        .collect();
    if !non_room.is_empty() {
        let map = level.build_map();
        let (sx, sy, sz) = level.size;
        let mut worst_count = 0usize;
        let mut worst_cell = (0i32, 0i32, 0i32);
        for x in 0..sx as i32 {
            for y in 0..sy as i32 {
                for z in 0..sz as i32 {
                    if map.cell_at(x, y, z).is_solid() {
                        continue;
                    }
                    let p = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.6];
                    let count = non_room
                        .iter()
                        .filter(|f| static_weight_at(f, p) > OVERLAP_WEIGHT_THRESHOLD)
                        .count();
                    if count > worst_count {
                        worst_count = count;
                        worst_cell = (x, y, z);
                    }
                }
            }
        }
        if worst_count > OVERLAP_MAX {
            warnings.push(format!(
                "{worst_count} non-room fields simultaneously exceed weight {OVERLAP_WEIGHT_THRESHOLD} \
                 at a walkable cell near {worst_cell:?} (more than the {OVERLAP_MAX}-field \
                 comfort ceiling before the top-3 selection + hysteresis even applies)"
            ));
            locations.push(Location::cell(worst_cell));
        }
    }

    if warnings.is_empty() {
        return Check::new(
            "fields",
            Severity::Pass,
            format!(
                "{} sound field(s) authored, no density/proximity/coverage concerns",
                fields.len()
            ),
        );
    }
    Check::new("fields", Severity::Warn, warnings.join("; ")).at_all(locations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_level::load_string;

    const HEADER: &str = "\
[meta]
name = L1-test
[room]
size = 20 20 10
[pedestal]
center = 1,1,1
volume = 1 1 1
[door]
pos = 17 6 1
[player]
spawn = 2 2 1
";

    fn load(extra: &str) -> Level {
        load_string(&format!("{HEADER}{extra}"), "test.lvl").expect("level parses")
    }

    #[test]
    fn no_fields_is_a_clean_pass() {
        let lvl = load("");
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Pass);
    }

    #[test]
    fn one_well_placed_field_is_a_clean_pass() {
        let lvl = load("[field exit]\nkind = exit_detune\nat = 17, 6, 1\n");
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    }

    #[test]
    fn campaign_level_with_no_exit_detune_door_warns() {
        let lvl = load("[field river]\nkind = lethe_wash\nat = 4, 9, 1\n");
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Warn);
        assert!(c.detail.contains("exit_detune"), "{}", c.detail);
    }

    #[test]
    fn non_campaign_level_with_no_exit_detune_is_fine() {
        let text = HEADER.replace("L1-test", "A1-demo")
            + "[field river]\nkind = lethe_wash\nat = 4, 9, 1\n";
        let lvl = load_string(&text, "test.lvl").expect("level parses");
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    }

    #[test]
    fn tail_f_next_to_the_door_warns() {
        let lvl = load("[field t]\nkind = tail_f\nat = 17, 6, 1\n");
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Warn);
        assert!(c.detail.contains("tail_f"), "{}", c.detail);
        assert!(!c.locations.is_empty());
    }

    #[test]
    fn tail_f_far_from_the_solve_path_is_fine() {
        let lvl = load(
            "[field exit]\nkind = exit_detune\nat = 17, 6, 1\n\
             [field t]\nkind = tail_f\nradius = 2\nat = 19, 19, 9\n",
        );
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    }

    #[test]
    fn more_than_two_hold_fields_warns() {
        let lvl = load(
            "[field exit]\nkind = exit_detune\nat = 17, 6, 1\n\
             [field a]\nkind = mnemos_hold\nat = 1,1,1\n\
             [field b]\nkind = mnemos_hold\nat = 2,2,1\n\
             [field c]\nkind = tail_f\nat = 3,3,1\n",
        );
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Warn);
        assert!(c.detail.contains("Hold-using"), "{}", c.detail);
    }

    #[test]
    fn dense_overlap_at_one_cell_warns() {
        // Four overlapping full-room-ish spheres centred on the same cell.
        let mut extra = String::from("[field e]\nkind = exit_detune\nat = 17, 6, 1\n");
        for (i, kind) in ["bit_rot", "watcher_gaze", "lure_wobble", "steam_warmth"]
            .iter()
            .enumerate()
        {
            extra.push_str(&format!(
                "[field f{i}]\nkind = {kind}\nat = 10, 10, 1\nradius = 8\n"
            ));
        }
        let lvl = load(&extra);
        let c = check_fields(&lvl);
        assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
        assert!(c.detail.contains("walkable cell"), "{}", c.detail);
    }
}
