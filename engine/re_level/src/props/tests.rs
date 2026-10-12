//! Prop-kit tests: per-kind footprints under all four facings, material packs,
//! parser errors (line + key), keep-out refusal, stair climbability and the
//! built-map stamping contract.

use super::*;
use crate::{load_string, Level};
use re_world::{CameraPose, PlayerMotion, PLAYER_EYE_HEIGHT};

/// A 24×24×10 room whose puzzle (nodes, pipes, switches, pedestal, door) sits in
/// the far strip `y ≥ 19`, leaving the middle free for props.
const ROOM: &str = "[meta]
name = prop-test
title = prop test

[room]
size = 24 24 10
ambient = solid
storey_height = 4
power_budget = 2
render_pipes = true

[node R]
pos = 2,22,2
glyph = R
color = #FF4040
is_input = yes

[node output]
pos = 21,22,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1

[pipe R->output]
state = ON
path = (2,22,2) (2,20,2) (21,20,2) (21,22,2)
switch_at = (6,20,2)

[inputs]
R = 1

[pedestal]
center = 18 22 2
volume = 1 1 1

[door]
pos = 22 1 1
target_parts = output

[player]
spawn = 12 17 1
facing = -y

";

fn level(props: &str) -> Level {
    let txt = format!("{ROOM}{props}");
    load_string(&txt, "prop-test.lvl").unwrap_or_else(|e| panic!("level must load: {e}\n{props}"))
}

fn load_err(props: &str) -> crate::ParseError {
    let txt = format!("{ROOM}{props}");
    match load_string(&txt, "prop-test.lvl") {
        Ok(_) => panic!("expected a parse error for:\n{props}"),
        Err(e) => e,
    }
}

/// 1-based line of the first line of `props` inside the full level text.
fn first_prop_line() -> usize {
    ROOM.lines().count() + 1
}

fn expand(lvl: &Level) -> PropExpansion {
    let (exp, errs) = expand_level(lvl, &lvl.build_map_base());
    assert!(errs.is_empty(), "unexpected expansion errors: {errs:?}");
    exp
}

fn prop(kind: &str, body: &str) -> String {
    format!("[prop p]\nkind = {kind}\n{body}\n")
}

/// A local-frame key for one cell: `(u, v, w, material)` with
/// `world = at + u·r − v·f + w·ẑ` inverted.
fn local_cell(
    at: (i32, i32, i32),
    f: Facing,
    c: &PropCell,
) -> (i32, i32, i32, Option<String>, String) {
    let (fv, rv) = (f.vec(), f.right());
    let d = (c.pos.0 - at.0, c.pos.1 - at.1, c.pos.2 - at.2);
    let u = d.0 * rv.0 + d.1 * rv.1;
    let v = -(d.0 * fv.0 + d.1 * fv.1);
    (
        u,
        v,
        d.2,
        c.mat.clone(),
        format!("{:?}", c.shape).chars().take(6).collect(),
    )
}

/// A local-frame key for one line point, rounded to 1/1000 of a cell.
fn local_point(at: (i32, i32, i32), f: Facing, p: Vec3) -> (i64, i64, i64) {
    let (fv, rv) = (f.vec(), f.right());
    let o = Vec3::new(at.0 as f32 + 0.5, at.1 as f32 + 0.5, at.2 as f32)
        + Vec3::new(fv.0 as f32, fv.1 as f32, 0.0) * 0.5
        - Vec3::new(rv.0 as f32, rv.1 as f32, 0.0) * 0.5;
    let d = p - o;
    let u = d.x * rv.0 as f32 + d.y * rv.1 as f32;
    let v = -(d.x * fv.0 as f32 + d.y * fv.1 as f32);
    let r = |x: f32| (x * 1000.0).round() as i64;
    (r(u), r(v), r(d.z))
}

type Footprint = (
    Vec<(i32, i32, i32, Option<String>, String)>,
    Vec<Vec<(i64, i64, i64)>>,
    usize,
    usize,
);

fn footprint(kind: &str, body: &str, at: (i32, i32, i32), f: Facing) -> Footprint {
    let lvl = level(&prop(
        kind,
        &format!(
            "at = {} {} {}\nfacing = {}\n{body}",
            at.0,
            at.1,
            at.2,
            f.as_str()
        ),
    ));
    let exp = expand(&lvl);
    assert!(
        exp.issues.iter().all(|i| i.level != IssueLevel::Fail),
        "{kind} facing {}: FAIL issue {:?}",
        f.as_str(),
        exp.issues
    );
    let mut cells: Vec<_> = exp.cells.iter().map(|c| local_cell(at, f, c)).collect();
    cells.sort();
    let mut lines: Vec<Vec<_>> = exp
        .lines
        .iter()
        .map(|l| l.points.iter().map(|p| local_point(at, f, *p)).collect())
        .collect();
    // Lines are compared up to a translation: float-anchored kinds (overgrowth)
    // pivot on their authored point rather than a cell corner. The cell pivot is
    // pinned exactly by the cell comparison and `rack_front_faces_facing`.
    let min = lines
        .iter()
        .flatten()
        .fold((i64::MAX, i64::MAX), |m, p| (m.0.min(p.0), m.1.min(p.1)));
    for l in &mut lines {
        for p in l.iter_mut() {
            p.0 -= min.0;
            p.1 -= min.1;
        }
    }
    lines.sort();
    (cells, lines, exp.holos.len(), exp.parts.len())
}

/// Every oriented kind, in open floor space: the stamp is non-empty and the SAME
/// shape in the prop's own frame under all four facings (rotation-exact, with
/// identical materials), i.e. `facing` rotates the prefab and nothing else.
#[test]
fn every_oriented_kind_rotates_exactly_under_all_four_facings() {
    let at = (11, 10, 1);
    let high = (11, 10, 4);
    let cases: &[(&str, &str, (i32, i32, i32))] = &[
        ("rack", "size = 1 2 3\npower = node:R", at),
        (
            "rack_row",
            "size = 6 1 2\ngap = 4\ndead = 0.3\ntray = yes",
            at,
        ),
        ("console", "size = 3 2 1\nmonitor = yes\nchair = yes", at),
        ("shelving", "size = 5 1 3\nhatch = /\nfaces = both", at),
        ("locker_bank", "size = 5 1 2\nopen = 3", at),
        ("tank", "size = 3 2 3\nemitter = yes", at),
        ("column", "size = 2 2 3\nribs = yes", at),
        ("partition", "size = 6 1 2\ndoor_gap = 2..3", at),
        ("debris", "size = 6 4 1\ncount = 9", at),
        ("stair", "form = straight\nrise = 4", at),
        ("gallery", "size = 5 3 1", high),
        (
            "holo_display",
            "size = 3 3 3\nshape = pyramid\nbase = yes\nstyle = 0",
            at,
        ),
        ("overgrowth", "size = 4 1 3\nfloor = yes", at),
    ];
    for &(kind, body, at) in cases {
        let reference = footprint(kind, body, at, Facing::PosY);
        assert!(
            !reference.0.is_empty() || !reference.1.is_empty(),
            "{kind}: stamped nothing"
        );
        for f in [Facing::PosX, Facing::NegX, Facing::NegY] {
            let got = footprint(kind, body, at, f);
            assert_eq!(
                got.0,
                reference.0,
                "{kind}: cells differ in the local frame at facing {}",
                f.as_str()
            );
            assert_eq!(
                got.1,
                reference.1,
                "{kind}: lines differ in the local frame at facing {}",
                f.as_str()
            );
            assert_eq!(
                (got.2, got.3),
                (reference.2, reference.3),
                "{kind}: holo/part counts differ at facing {}",
                f.as_str()
            );
        }
    }
}

/// Exact footprints of the box-like kinds: the cell set is the authored
/// `W × D × H` box (rotated), nothing more.
#[test]
fn box_kinds_fill_exactly_their_size() {
    let at = (11, 10, 1);
    for f in Facing::ALL {
        for (kind, body, (w, d, h)) in [
            ("shelving", "size = 4 1 3", (4, 1, 3)),
            ("tank", "size = 2 2 3", (2, 2, 3)),
            ("column", "size = 2 2 3", (2, 2, 3)),
            ("partition", "size = 2 1 2", (2, 1, 2)),
        ] {
            let lvl = level(&prop(
                kind,
                &format!("at = 11 10 1\nfacing = {}\n{body}", f.as_str()),
            ));
            let exp = expand(&lvl);
            let mut got: Vec<(i32, i32, i32)> = exp
                .cells
                .iter()
                .map(|c| {
                    let k = local_cell(at, f, c);
                    (k.0, k.1, k.2)
                })
                .collect();
            got.sort_unstable();
            got.dedup();
            let mut want = Vec::new();
            for u in 0..w {
                for v in 0..d {
                    for z in 0..h {
                        want.push((u, v, z));
                    }
                }
            }
            want.sort();
            assert_eq!(got, want, "{kind} facing {}", f.as_str());
        }
    }
}

/// The rack is a 1×2×3 body: the back row (v = 1) is solid full height, the
/// front row (v = 0) is the recess — a trim cap on top, the blade slot at eye
/// height, walkable air at the floor — and nothing is stamped in front of it.
#[test]
fn rack_front_faces_facing() {
    for f in Facing::ALL {
        let lvl = level(&prop(
            "rack",
            &format!(
                "at = 11 10 1
facing = {}
size = 1 2 3
power = node:R",
                f.as_str()
            ),
        ));
        let exp = expand(&lvl);
        let fv = f.vec();
        let has = |p: Cell3| exp.cells.iter().any(|c| c.pos == p);
        for z in 1..=3 {
            assert!(
                has((11 - fv.0, 10 - fv.1, z)),
                "facing {}: back row z={z}",
                f.as_str()
            );
        }
        assert!(has((11, 10, 3)), "facing {}: trim cap", f.as_str());
        assert!(
            !has((11, 10, 1)),
            "facing {}: the recess floor stays open",
            f.as_str()
        );
        assert!(
            exp.parts.iter().any(|p| p.pos == (11, 10, 2)),
            "facing {}: blade slot",
            f.as_str()
        );
        assert!(
            !has((11 + fv.0, 10 + fv.1, 1)),
            "facing {}: nothing in front",
            f.as_str()
        );
    }
    // A rack whose front faces a wall is refused.
    let e = load_err(
        "[prop p]
kind = rack
at = 11 1 1
facing = -y
size = 1 2 3
",
    );
    assert!(
        e.msg.contains("[prop p]") && e.msg.contains("facing"),
        "{e}"
    );
}

/// Non-oriented kinds expand deterministically and reject `facing`.
#[test]
fn non_oriented_kinds_expand_and_ignore_facing() {
    let cases: &[(&str, &str)] = &[
        ("cctv", "at = 1 1 6\nlook = 8 9 2"),
        ("cable_tray", "follow = pipe:R->output\nlift = 2"),
        (
            "conduit",
            "at = 3 4 3\nto = 12 4 3\nriser = both\nend = vent",
        ),
        ("cable_run", "follow = pipe:R->output\nbundle = 2"),
        ("cable_drop", "at = 8.5 4.5 9\nto = 8.5 4.5 3\nstrands = 3"),
        (
            "lamp_pendant",
            "at = 8.5 6.5 4.8\nlight = yes\nshape = cone",
        ),
        ("budget_bank", "at = 11 0 4\nfacing = +y"),
        (
            "wall_panel",
            "at = 8 0 1\nfacing = +y\nsize = 4 1 3\nmode = bank\nscreen = mixed",
        ),
    ];
    for &(kind, body) in cases {
        let a = expand(&level(&prop(kind, body)));
        let b = expand(&level(&prop(kind, body)));
        assert!(
            !a.cells.is_empty() || !a.lines.is_empty(),
            "{kind}: stamped nothing"
        );
        assert_eq!(a.cells, b.cells, "{kind}: cells must be deterministic");
        assert_eq!(a.lines, b.lines, "{kind}: lines must be deterministic");
    }
    let hp = "at = 4 3 1
size = 6 4 1";
    let a = expand(&level(&prop("hanging_panels", hp)));
    assert!(!a.cells.is_empty(), "hanging_panels: stamped nothing");
    assert_eq!(a.cells, expand(&level(&prop("hanging_panels", hp))).cells);
    for kind in ["cctv", "cable_drop", "lamp_pendant", "conduit"] {
        let body = cases.iter().find(|c| c.0 == kind).unwrap().1;
        let lvl = level(&prop(kind, &format!("{body}\nfacing = +x")));
        assert!(
            lvl.unknown_keys
                .iter()
                .any(|k| k.key == "facing" && k.section == "prop"),
            "{kind}: facing must be reported as an unused key"
        );
    }
}

/// `style =` selects the §3.5 pack; every synthesized pack material is that
/// pack's, byte-identical to `pack_material`, and `mat.SLOT` overrides it.
#[test]
fn material_pack_is_applied_and_overridable() {
    for style in [
        PropStyle::Lab,
        PropStyle::Range,
        PropStyle::Vault,
        PropStyle::Archive,
        PropStyle::Dead,
        PropStyle::Void,
    ] {
        let lvl = level(&prop(
            "shelving",
            &format!(
                "at = 11 10 1\nfacing = +y\nsize = 4 1 4\nstyle = {}",
                style.as_str()
            ),
        ));
        let exp = expand(&lvl);
        let prefix = format!("{PROP_MATERIAL_PREFIX}{}_", style.as_str());
        let used: Vec<&str> = exp.cells.iter().filter_map(|c| c.mat.as_deref()).collect();
        assert!(!used.is_empty());
        for m in &used {
            if m.starts_with(PROP_MATERIAL_PREFIX)
                && !m.starts_with(&format!("{PROP_MATERIAL_PREFIX}c_"))
                && !m.starts_with(&format!("{PROP_MATERIAL_PREFIX}p_"))
            {
                assert!(
                    m.starts_with(&prefix),
                    "{}: material {m} is not from the pack",
                    style.as_str()
                );
            }
        }
        let body = pack_material(style, MatSlot::Body);
        let synth = exp
            .materials
            .iter()
            .find(|m| m.name == body.name)
            .expect("pack body synthesized");
        assert_eq!(
            (synth.fg, synth.bg, &synth.preset),
            (body.fg, body.bg, &body.preset)
        );
        assert!(
            used.contains(&body.name.as_str()),
            "{}: body slot not used",
            style.as_str()
        );
    }
    // A hex override replaces the body slot with a flat material.
    let lvl = level(&prop(
        "shelving",
        "at = 11 10 1\nfacing = +y\nsize = 4 1 4\nmat.body = #FF0000",
    ));
    let exp = expand(&lvl);
    assert!(exp
        .cells
        .iter()
        .any(|c| c.mat.as_deref() == Some("__prop_c_ff0000")));
    assert!(!exp
        .cells
        .iter()
        .any(|c| c.mat.as_deref()
            == Some(pack_material(PropStyle::Lab, MatSlot::Body).name.as_str())));
    // An authored material name is used verbatim and not re-synthesized.
    let lvl = level(&format!(
        "[material steel]\nfg = #C0C0C0\nbg = #101010\n\n{}",
        prop(
            "shelving",
            "at = 11 10 1\nfacing = +y\nsize = 4 1 4\nmat.body = steel"
        )
    ));
    let exp = expand(&lvl);
    assert!(exp.cells.iter().any(|c| c.mat.as_deref() == Some("steel")));
    assert!(!exp.materials.iter().any(|m| m.name == "steel"));
}

/// The built map carries the prop cells with the right palette ids, and a level
/// with no props builds byte-identically to `build_map_base`.
#[test]
fn build_map_stamps_props_and_is_identity_without_them() {
    let plain = level("");
    assert_eq!(
        crate::CellGrid::from_map(&plain.build_map()),
        crate::CellGrid::from_map(&plain.build_map_base())
    );

    let lvl = level(&prop("shelving", "at = 11 10 1\nfacing = +y\nsize = 4 1 3"));
    let exp = expand(&lvl);
    let map = lvl.build_map();
    let grid = crate::CellGrid::from_map(&map);
    for c in &exp.cells {
        assert_eq!(
            map.cell_at(c.pos.0, c.pos.1, c.pos.2),
            c.kind,
            "{:?}",
            c.pos
        );
        if let Some(m) = &c.mat {
            let id = exp.palette_id(&lvl, m);
            assert!(id > 0);
            assert!(
                grid.material_ids.contains(&(c.pos.0, c.pos.1, c.pos.2, id)),
                "{:?} must carry palette id {id}",
                c.pos
            );
        }
    }
}

// ───────────────────────────── parser errors ─────────────────────────────────

#[test]
fn parser_errors_carry_line_and_key() {
    let l0 = first_prop_line();
    // Unknown kind: points at the `kind` line.
    let e = load_err("[prop p]\nkind = sofa\nat = 3 3 1\n");
    assert_eq!(e.line, l0 + 1, "{e}");
    assert!(
        e.msg.contains("[prop p]") && e.msg.contains("kind") && e.msg.contains("sofa"),
        "{e}"
    );

    // Bad facing: points at the `facing` line.
    let e = load_err("[prop p]\nkind = rack\nat = 3 3 1\nfacing = up\n");
    assert_eq!(e.line, l0 + 3, "{e}");
    assert!(e.msg.contains("facing"), "{e}");

    // Size below the kind minimum (above the maximum is clamped with a warning).
    let e = load_err("[prop p]\nkind = rack\nat = 3 3 1\nfacing = +y\nsize = 1 0 3\n");
    assert_eq!(e.line, l0 + 4, "{e}");
    assert!(e.msg.contains("size"), "{e}");

    // Missing `kind` / `at`: point at the header.
    let e = load_err("[prop p]\nat = 3 3 1\n");
    assert_eq!(e.line, l0, "{e}");
    assert!(e.msg.contains("missing required key 'kind'"), "{e}");
    let e = load_err("[prop p]\nkind = rack\n");
    assert_eq!(e.line, l0, "{e}");
    assert!(e.msg.contains("missing required key 'at'"), "{e}");

    // Solid kinds need integer cells.
    let e = load_err("[prop p]\nkind = rack\nat = 3.5 3 1\n");
    assert_eq!(e.line, l0 + 2, "{e}");
    assert!(e.msg.contains("at") && e.msg.contains("integer"), "{e}");

    // Duplicate key.
    let e = load_err("[prop p]\nkind = rack\nat = 3 3 1\nat = 4 4 1\n");
    assert_eq!(e.line, l0 + 3, "{e}");
    assert!(e.msg.contains("duplicate"), "{e}");

    // Bad power source.
    let e = load_err("[prop p]\nkind = rack\nat = 3 3 1\npower = node:NOPE\n");
    assert_eq!(e.line, l0 + 3, "{e}");
    assert!(e.msg.contains("power"), "{e}");

    // Header: empty or malformed name.
    let e = load_err("[prop]\nkind = rack\nat = 3 3 1\n");
    assert_eq!(e.line, l0, "{e}");
    let e = load_err("[prop Big-Name]\nkind = rack\nat = 3 3 1\n");
    assert_eq!(e.line, l0, "{e}");
    assert!(e.msg.contains("[a-z0-9_]"), "{e}");

    // An unknown `mat.` slot is not a key at all: reported, not fatal.
    let lvl = level(
        "[prop p]
kind = rack
at = 3 3 1
mat.chrome = #FFFFFF
",
    );
    assert!(lvl
        .unknown_keys
        .iter()
        .any(|k| k.key == "mat.chrome" && k.line == l0 + 3));
    // A known slot with an unresolvable value is an error on that line.
    let e = load_err(
        "[prop p]
kind = rack
at = 3 3 1
mat.face = no_such_material
",
    );
    assert_eq!(e.line, l0 + 3, "{e}");
    assert!(e.msg.contains("mat.face"), "{e}");
}

#[test]
fn unknown_and_inapplicable_keys_are_reported_not_fatal() {
    let lvl = level("[prop p]\nkind = rack\nat = 3 3 1\nfacing = +y\nwobble = 3\nribs = yes\n");
    let keys: Vec<(&str, usize)> = lvl
        .unknown_keys
        .iter()
        .filter(|k| k.section == "prop")
        .map(|k| (k.key.as_str(), k.line))
        .collect();
    let l0 = first_prop_line();
    assert!(keys.contains(&("wobble", l0 + 4)), "{keys:?}");
    assert!(keys.contains(&("ribs", l0 + 5)), "{keys:?}");
}

#[test]
fn every_kind_name_round_trips() {
    for k in PropKind::ALL {
        assert_eq!(PropKind::parse(k.as_str()), Some(k));
    }
    assert_eq!(PropKind::ALL.len(), 24);
    for f in Facing::ALL {
        assert_eq!(Facing::parse(f.as_str()), Some(f));
    }
}

// ─────────────────────────────── keep-out ────────────────────────────────────

/// A solid prop on a puzzle cell is refused with an error naming both.
#[test]
fn solid_prop_on_a_puzzle_cell_is_refused_naming_both() {
    // The switch column at (6,20,*).
    let e = load_err("[prop blocker]\nkind = column\nat = 6 20 1\nsize = 1 1 3\n");
    assert!(e.msg.contains("[prop blocker]"), "{e}");
    assert!(e.msg.contains("switch"), "{e}");
    assert_eq!(e.line, first_prop_line(), "{e}");
    // The pipe path at y=20.
    let e = load_err("[prop blocker]\nkind = shelving\nat = 10 20 2\nfacing = -y\nsize = 2 1 2\n");
    assert!(
        e.msg.contains("[prop blocker]") && e.msg.contains("pipe"),
        "{e}"
    );
    // The door.
    let e = load_err("[prop blocker]\nkind = column\nat = 22 1 1\nsize = 1 1 2\n");
    assert!(
        e.msg.contains("[prop blocker]") && e.msg.contains("door"),
        "{e}"
    );
}

/// Non-solid dressing (lines, a cable run) may cross the puzzle freely.
#[test]
fn line_kinds_may_cross_the_puzzle() {
    let lvl = level(&prop("cable_run", "follow = pipe:R->output"));
    let exp = expand(&lvl);
    assert!(exp.cells.is_empty());
    assert!(!exp.lines.is_empty());
}

// ──────────────────────────────── stairs ─────────────────────────────────────

/// The straight stair the kit emits is climbable with the real gravity step:
/// walking its treads raises the eye by the full rise and stays grounded.
#[test]
fn straight_stair_is_climbable_with_step_gravity() {
    let lvl = level(&prop(
        "stair",
        "at = 11 4 1\nfacing = +y\nform = straight\nrise = 4",
    ));
    let exp = expand(&lvl);
    assert!(!exp.stair_cells.is_empty());
    let map = lvl.build_map();
    // Walk the emitted treads from the lowest to the highest,
    // in the direction the run climbs.
    let mut treads: Vec<Cell3> = exp.stair_cells;
    treads.sort_by_key(|c| c.2);
    let first = treads[0];
    let last = *treads.last().unwrap();
    assert_eq!(
        last.2 - first.2,
        3,
        "rise 4 = 4 treads/landing over 4 z levels: {treads:?}"
    );
    let (dx, dy) = ((last.0 - first.0).signum(), (last.1 - first.1).signum());
    let mut cam = CameraPose {
        pos: Vec3::new(
            first.0 as f32 + 0.5 - dx as f32 * 0.9,
            first.1 as f32 + 0.5 - dy as f32 * 0.9,
            first.2 as f32 + PLAYER_EYE_HEIGHT,
        ),
        yaw: 0.0,
        pitch: 0.0,
    };
    let mut motion = PlayerMotion {
        vz: 0.0,
        grounded: false,
        ..Default::default()
    };
    for _ in 0..30 {
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
    }
    assert!(motion.grounded, "starts grounded");
    let z0 = cam.pos.z;
    let steps = ((last.0 - first.0).abs() + (last.1 - first.1).abs() + 1) * 20;
    for _ in 0..steps {
        cam.pos.x += dx as f32 * 0.05;
        cam.pos.y += dy as f32 * 0.05;
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }
    assert!(motion.grounded, "still grounded at the top");
    let rise = cam.pos.z - z0;
    assert!(
        rise > 3.0,
        "the stair must lift the player ~4 cells, rose {rise:.2}"
    );
}

#[test]
fn spiral_stair_bad_rise_names_the_nearest_valid_rise() {
    let e =
        load_err("[prop sp]\nkind = stair\nat = 11 10 1\nfacing = +y\nform = spiral\nrise = 5\n");
    assert!(e.msg.contains("[prop sp]") && e.msg.contains("rise"), "{e}");
}

// ──────────────────────────────── power ─────────────────────────────────────

/// Power bindings are parsed and recorded as gates the game can drive.
#[test]
fn power_bindings_are_recorded_as_gates() {
    let lvl = level(&prop(
        "rack",
        "at = 11 10 1\nfacing = +y\nsize = 1 2 3\npower = node:R",
    ));
    assert_eq!(lvl.prop_power.len(), 1, "{:?}", lvl.prop_power);
    let g = &lvl.prop_power[0];
    assert_eq!(g.prop, "p");
    assert!(
        !g.parts.is_empty() || !g.lines.is_empty(),
        "the gate drives something"
    );
    // An input node with value 1 starts lit.
    assert!(g.initial_lit);

    let lvl = level(&prop(
        "rack",
        "at = 11 10 1\nfacing = +y\nsize = 1 2 3\npower = node:R\ninvert = yes",
    ));
    assert!(
        !lvl.prop_power[0].initial_lit,
        "invert flips the initial state"
    );

    let lvl = level(&prop(
        "rack",
        "at = 11 10 1\nfacing = +y\nsize = 1 2 3\npower = none",
    ));
    assert!(lvl.prop_power.is_empty());
}

// ─────────────────────────────── notes ──────────────────────────────────────

/// A notice board adopts its note: the note moves onto the board and its
/// authored position is kept for the editor's round trip.
#[test]
fn notice_board_adopts_its_note_and_keeps_the_origin() {
    let lvl = level(
        "[prop board]\nkind = notice_board\nat = 8 1 1\nfacing = +y\nsize = 2 1 2\nnote = system_log\n\n\
         [note system_log pos=12,6,2 facing=+y]\n    NOTICE.\n\n",
    );
    assert_eq!(lvl.prop_note_origins.len(), 1);
    let o = &lvl.prop_note_origins[0];
    assert_eq!((o.x, o.y, o.z), (12, 6, 2));
    let n = &lvl.notes[o.note];
    assert_ne!((n.x, n.y, n.z), (12, 6, 2), "the note moved onto the board");
    assert!(
        (8..=9).contains(&n.x) && n.y == 1,
        "note at ({},{},{})",
        n.x,
        n.y,
        n.z
    );
}

// ─────────────────────────────── strip ──────────────────────────────────────

/// `strip_prop_cells` reverts every prop cell of a built grid to the base, so an
/// editor save does not bake prop geometry into `[cells]`.
#[test]
fn strip_prop_cells_reverts_to_the_base_grid() {
    let lvl = level(&prop("shelving", "at = 11 10 1\nfacing = +y\nsize = 4 1 3"));
    let built = crate::CellGrid::from_map(&lvl.build_map());
    let base = crate::CellGrid::from_map(&lvl.build_map_base());
    assert_ne!(built, base);
    let mut stripped = built;
    strip_prop_cells(&lvl, &mut stripped);
    assert_eq!(stripped, base);
}

/// `furniture_trace` takes `at` as the min corner of its world footprint: under
/// every facing the outline stays inside `[at, at + extent]` (W and D swap on the
/// x-facings), is the same stroke set, and reaches every edge of that box.
#[test]
fn furniture_trace_footprint_rotates_inside_its_box() {
    let mut counts = Vec::new();
    for f in Facing::ALL {
        let lvl = level(&prop(
            "furniture_trace",
            &format!(
                "shape = table\nat = 6.5 4.5 1.0\nfacing = {}\nsize = 2 1",
                f.as_str()
            ),
        ));
        let exp = expand(&lvl);
        assert!(exp.cells.is_empty(), "furniture is lines only");
        let (ex, ey) = if f.vec().0 != 0 {
            (1.0, 2.0)
        } else {
            (2.0, 1.0)
        };
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in exp.lines.iter().flat_map(|l| l.points.iter()) {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
        assert!(
            near(lo.x, 6.5) && near(lo.y, 4.5),
            "facing {}: lo {lo}",
            f.as_str()
        );
        assert!(
            near(hi.x, 6.5 + ex) && near(hi.y, 4.5 + ey),
            "facing {}: hi {hi}",
            f.as_str()
        );
        assert!(
            near(lo.z, 1.0) && near(hi.z, 1.0 + 0.45 * TRACE_H),
            "facing {}: table top at 0.45 body units",
            f.as_str()
        );
        counts.push(exp.lines.iter().map(|l| l.points.len()).collect::<Vec<_>>());
    }
    assert!(counts.windows(2).all(|w| w[0] == w[1]), "{counts:?}");
}

// ─────────────────────────── lit slots (2026-09-24) ─────────────────────────

/// `mat.lit` is the gate's colour: a `door` gate (which has no puzzle hue of its
/// own) lights in the authored colour, and that colour is what the gate reports as
/// `lit_rgb` — the hue lamps, emitters and lvlcheck's colour law read.
#[test]
fn mat_lit_sets_the_gate_colour_for_door_and_seq() {
    let lvl = level(&prop(
        "rack",
        "at = 11 10 1\nfacing = +y\nsize = 1 2 3\npower = door\nmat.lit = #FF4040",
    ));
    assert_eq!(lvl.prop_power.len(), 1);
    assert_eq!(
        lvl.prop_power[0].lit_rgb & 0xFF_FFFF,
        0xFF_4040,
        "mat.lit is the door gate's colour"
    );

    // A named authored material: its fg is the colour.
    let lvl = level(&format!(
        "[material red_lit]\npreset = neon\nfg = #FF4040\nbg = #B02020\n\n{}",
        prop(
            "rack",
            "at = 11 10 1\nfacing = +y\nsize = 1 2 3\npower = door\nmat.lit = red_lit"
        )
    ));
    assert_eq!(lvl.prop_power[0].lit_rgb & 0xFF_FFFF, 0xFF_4040);

    // Without `mat.lit` a door gate keeps the pack colour (unchanged behaviour).
    let lvl = level(&prop(
        "rack",
        "at = 11 10 1\nfacing = +y\nsize = 1 2 3\npower = door",
    ));
    assert_eq!(lvl.prop_power[0].lit_rgb, pack_lit_rgb(PropStyle::Lab));
}

/// A `rack_row` is one deep, so its powered units carry their lit blade IN the face:
/// a gated row has R1 lit cells, one per powered unit, and an unpowered row none.
#[test]
fn a_powered_rack_row_has_lit_cells() {
    let lvl = level(&prop(
        "rack_row",
        "at = 8 10 1\nfacing = +y\nsize = 3 1 3\npower = node:R",
    ));
    let exp = expand(&lvl);
    assert_eq!(
        exp.parts.len(),
        3,
        "one lit blade per unit: {:?}",
        exp.parts
    );
    assert!(
        exp.parts.iter().all(|p| p.pos.2 == 2),
        "the blade is the unit's second cell"
    );
    let lvl = level(&prop("rack_row", "at = 8 10 1\nfacing = +y\nsize = 3 1 3"));
    assert!(expand(&lvl).parts.is_empty(), "no power, no lit slot");
}

/// The kit's lit screen backs its bands with half its hue, not a near-black 12 %:
/// a powered slot must be brighter than a lamp-lit wall around it.
#[test]
fn the_default_lit_screen_has_a_lit_backing() {
    let m = lit_screen_material(0x40_FF80);
    let (r, g, b) = ((m.bg >> 16) & 0xFF, (m.bg >> 8) & 0xFF, m.bg & 0xFF);
    assert!(
        g >= 0x70,
        "backing {:#08x} must be a lit half of the hue",
        m.bg
    );
    assert!(r < g && b < g, "the backing keeps the hue");
}

// ─────────────────────── line heights are world z (2026-09-24) ──────────────

use kinds::{RAIL_H, TRACE_H};

/// The standing eye is 0.5 above the floor top in world z (`[line]` z is world z).
const EYE_ABOVE_FLOOR: f32 = 0.5;

/// Trace furniture and rails sit BELOW the eye in world z: a chair back, a rail and a
/// gallery handrail. They were authored when the eye was 1.0 above the floor, and
/// drawn in that convention a rail at 0.9 stood at almost twice eye height.
#[test]
fn chair_backs_and_rails_stay_below_the_eye() {
    let top = |props: &str| -> f32 {
        expand(&level(props))
            .lines
            .iter()
            .flat_map(|l| l.points.iter())
            .map(|p| p.z)
            .fold(f32::MIN, f32::max)
    };
    let chair = top(&prop(
        "furniture_trace",
        "shape = chair\nat = 6.5 4.5 1.0\nfacing = +y\nsize = 0.6 0.6",
    ));
    assert!(
        chair > 1.0 + 0.3 && chair < 1.0 + EYE_ABOVE_FLOOR,
        "chair back top z {chair}"
    );
    assert!(
        (chair - (1.0 + 0.75 * TRACE_H)).abs() < 1e-4,
        "chair back at 0.75 body units: {chair}"
    );
    let rail = top(&prop(
        "furniture_trace",
        "shape = rail\nat = 6.5 4.5 1.0\nfacing = +y\nsize = 3 0.2",
    ));
    assert!(
        (rail - (1.0 + RAIL_H)).abs() < 1e-4 && rail < 1.0 + EYE_ABOVE_FLOOR,
        "rail z {rail}"
    );
    // A gallery deck stood on at z = 4 carries its handrail just under that eye.
    let exp = expand(&level(&prop(
        "gallery",
        "at = 11 10 4\nfacing = +y\nsize = 5 3 1",
    )));
    let deck_top = exp
        .cells
        .iter()
        .map(|c| c.pos.2 + 1)
        .max()
        .expect("a gallery deck") as f32;
    let hand = exp
        .lines
        .iter()
        .flat_map(|l| l.points.iter())
        .map(|p| p.z)
        .fold(f32::MIN, f32::max);
    assert!(
        hand > deck_top && hand < deck_top + EYE_ABOVE_FLOOR,
        "handrail z {hand} must stand under the eye above the deck top {deck_top}"
    );
}

// ──────────────── keep-out: notes, mirrors, portals (2026-09-24) ─────────────

fn expand_errs(lvl: &Level) -> Vec<String> {
    expand_level(lvl, &lvl.build_map_base())
        .1
        .into_iter()
        .map(|e| e.msg)
        .collect()
}

/// A note's reader space (reader cell → the wall the binder mounts the placard on)
/// is HARD keep-out: a solid prop there would become the "first masonry" on the
/// reader's gaze and silently steal the placard. The load refuses, naming the note.
#[test]
fn a_prop_in_a_notes_reader_space_is_refused() {
    let note = "[note system_log pos=6,4,1 facing=+y]\n    > the wall behind this reader.\n\n";
    // Reader at (6,4) looks -y at the perimeter wall y = 0: (6,1..=4,1) is its space.
    let e = load_err(&format!(
        "{note}{}",
        prop("rack", "at = 6 2 1\nfacing = +y\nsize = 1 1 1")
    ))
    .to_string();
    assert!(
        e.contains("note #0") && e.contains("reader space"),
        "a rack between the reader and its wall must be refused: {e}"
    );
    let clear = level(&format!(
        "{note}{}",
        prop("rack", "at = 9 2 1\nfacing = +y\nsize = 1 1 1")
    ));
    assert!(
        expand_errs(&clear).is_empty(),
        "a rack beside the reader space is fine"
    );
}

/// A `[wall_decal]` mirror — a patch re-kinded to a surface whose `[material]` is
/// reflective — is keep-out (face + the cell straight out) like a zone mirror: a solid
/// prop there is stamped but reported as a lvlcheck FAIL.
#[test]
fn a_prop_against_a_decal_mirror_is_reported() {
    let mirror = "[material glass]\nfg = #C8D2FF\nbg = #05060A\nreflective = true\nsurface = wall_matrix\n\n\
                  [wall_decal]\nsurface = matrix\nregion = (8,0,1)-(10,0,3)\n\n";
    let exp = expand(&level(&format!(
        "{mirror}{}",
        prop("rack", "at = 9 1 1\nfacing = +y\nsize = 1 1 1")
    )));
    assert!(
        exp.issues.iter().any(|i| i.level == IssueLevel::Fail
            && i.msg.contains("mirror decal")
            && i.msg.contains("glass")),
        "a rack in front of the mirror must be a lvlcheck FAIL naming it: {:?}",
        exp.issues.iter().map(|i| &i.msg).collect::<Vec<_>>()
    );
    // Diagonally past the mirror's end is not "in front" of it.
    let exp = expand(&level(&format!(
        "{mirror}{}",
        prop("rack", "at = 11 1 1\nfacing = +y\nsize = 1 1 1")
    )));
    assert!(
        exp.issues.iter().all(|i| !i.msg.contains("mirror")),
        "the diagonal cell is clear"
    );
    let errs = expand_errs(&level(&format!(
        "{mirror}{}",
        prop("rack", "at = 14 1 1\nfacing = +y\nsize = 1 1 1")
    )));
    assert!(
        errs.is_empty(),
        "a rack away from the mirror is fine: {errs:?}"
    );
}

/// `posts =` on a gallery: `none` stands no post, `no_ends` drops the end posts
/// that `auto` stands, and a list stands exactly those deck cells.
#[test]
fn gallery_posts_key() {
    let posts = |key: &str| -> Vec<Cell3> {
        let body = format!("at = 6 6 4\nfacing = +y\nsize = 9 2 1\n{key}");
        let mut v: Vec<Cell3> = expand(&level(&prop("gallery", &body)))
            .cells
            .iter()
            .map(|c| c.pos)
            .filter(|p| p.2 < 4)
            .collect();
        v.sort_unstable();
        v
    };
    let auto = posts("");
    let no_ends = posts("posts = no_ends");
    assert!(!auto.is_empty(), "auto stands posts");
    assert!(posts("posts = none").is_empty(), "none stands no post");
    assert!(
        !no_ends.is_empty() && no_ends.len() < auto.len(),
        "no_ends: {} vs auto {}",
        no_ends.len(),
        auto.len()
    );
    let listed = posts("posts = 0 0; 8 1");
    let cols: HashSet<(i32, i32)> = listed.iter().map(|p| (p.0, p.1)).collect();
    assert_eq!(cols.len(), 2, "exactly the two listed columns: {listed:?}");
    let e = load_err(&prop(
        "gallery",
        "at = 6 6 4\nfacing = +y\nsize = 9 2 1\nposts = 12 0",
    ))
    .to_string();
    assert!(e.contains("posts") && e.contains("outside"), "{e}");
}

/// A gallery post stands ON a prop built before it (here a rack under the deck
/// corner), not through it: the post column starts above the rack's top.
#[test]
fn gallery_posts_see_earlier_prop_cells() {
    let under = "[prop under]\nkind = rack\nat = 6 6 1\nfacing = +y\nsize = 1 1 2\n\n";
    let lvl = level(&format!(
        "{under}{}",
        prop(
            "gallery",
            "at = 6 6 4\nfacing = +y\nsize = 9 2 1\nposts = 0 0"
        )
    ));
    let exp = expand(&lvl);
    let rack_top = exp
        .cells
        .iter()
        .filter(|c| c.prop == 0)
        .map(|c| c.pos)
        .max_by_key(|p| p.2)
        .unwrap();
    let post: Vec<Cell3> = exp
        .cells
        .iter()
        .filter(|c| c.prop == 1 && c.pos.2 < 4)
        .map(|c| c.pos)
        .collect();
    let (px, py) = (post[0].0, post[0].1);
    assert_eq!(
        (px, py),
        (rack_top.0, rack_top.1),
        "the listed post sits on the rack's column"
    );
    assert!(
        post.iter().all(|p| p.2 > rack_top.2),
        "post {post:?} must start above the rack top {rack_top:?}"
    );
}

/// A floor-standing trace piece (table, chair, planter…) lays a dim contact-shadow
/// decal on the floor cells under its footprint, so it reads as an object from
/// spawn distance. The decal re-materials FLOOR geometry only — the piece itself
/// stays a non-colliding outline.
#[test]
fn floor_standing_trace_furniture_lays_a_base_decal_under_its_footprint() {
    let lvl = level(&prop(
        "furniture_trace",
        "shape = table\nat = 6.0 4.0 1.0\nfacing = +y\nsize = 2 1",
    ));
    let exp = expand(&lvl);
    assert_eq!(
        exp.decals.len(),
        1,
        "one base decal for one table: {:?}",
        exp.decals
    );
    let d = &exp.decals[0];
    assert_eq!(
        (d.min, d.max),
        ((6, 4, 0), (7, 4, 0)),
        "base covers the 2x1 footprint on the z=0 floor"
    );
    assert!(
        d.mat.starts_with(PROP_MATERIAL_PREFIX),
        "synthesized base material, got {}",
        d.mat
    );
    // A wall-hung frame is not floor-standing: no base.
    let frame = level(&prop(
        "furniture_trace",
        "shape = frame\nat = 6.0 4.0 2.0\nfacing = +y\nsize = 0.8 0.0 0.6",
    ));
    assert!(
        expand(&frame).decals.is_empty(),
        "a hung frame lays no floor base"
    );
    // And the decal really lands in the built map.
    let map = lvl.build_map();
    let base = lvl.build_map_base();
    assert_ne!(
        map.side_data_at(6, 4, 0),
        base.side_data_at(6, 4, 0),
        "floor cell under the table is re-materialled"
    );
}

// ─────────────────────── solid furniture (pass 2, item 4) ───────────────────────

/// `furniture_trace solid = yes`: one colliding `WallSolid` per covered cell, standing
/// in the walking cell on the floor, in a z-band material whose band ends below the
/// standing eye and carries a bright top. A sub-cell stool straddling a cell line is
/// ONE block; a 3.8 × 2.8 table covers the 4 × 3 cells whose centres it spans.
#[test]
fn solid_furniture_is_one_low_banded_block_per_covered_cell() {
    let stool = level(&prop(
        "furniture_trace",
        "shape = chair\nat = 6.6 4.6 1.0\nsize = 0.7 0.7\nsolid = yes",
    ));
    let exp = expand(&stool);
    assert_eq!(
        exp.cells.len(),
        1,
        "{:?}",
        exp.cells.iter().map(|c| c.pos).collect::<Vec<_>>()
    );
    let c = &exp.cells[0];
    assert_eq!((c.pos, c.kind), ((6, 4, 1), CellKind::WallSolid));
    let name = c.mat.clone().expect("a solid block carries its material");
    assert!(name.starts_with(PROP_MATERIAL_PREFIX), "{name}");
    assert!(
        name.contains("~z0_20~t"),
        "band 0..0.20 with a top tone: {name}"
    );
    assert!(!exp.lines.is_empty(), "the chair keeps its back strokes");
    assert!(
        stool.build_map().cell_at(6, 4, 1).is_solid(),
        "the block collides"
    );

    let table = level(&prop(
        "furniture_trace",
        "shape = table\nat = 6.1 5.1 1.0\nsize = 3.8 2.8\nsolid = yes",
    ));
    let mut cells: Vec<_> = expand(&table).cells.iter().map(|c| c.pos).collect();
    cells.sort_unstable();
    let mut want: Vec<_> = (5..=7)
        .flat_map(|y| (6..=9).map(move |x| (x, y, 1)))
        .collect();
    want.sort_unstable();
    assert_eq!(cells, want);

    for shape in SOLID_FURNITURE_SHAPES {
        let band = solid_furniture_band(shape);
        assert!(
            band > 0.1 && band < 0.5,
            "{shape}: {band} must sit under the eye (0.5)"
        );
        let m = solid_furniture_material(0x6C_7888, shape);
        assert!(m.name.ends_with(&zband_suffix(
            0.0,
            band,
            Some((brighten(0x6C_7888, 1.9), SOLID_FURNITURE_TOP_FROM))
        )));
        assert!(
            m.preset.is_none(),
            "a preset name would be re-encoded and lose the band"
        );
        let top = brighten(0x6C_7888, 1.9);
        let peak = |v: u32| ((v >> 16) & 0xFF).max((v >> 8) & 0xFF).max(v & 0xFF);
        assert!(
            peak(top) > peak(m.fg) + 60,
            "{shape}: the top is a BRIGHT rim over the body"
        );
    }
}

/// Without `solid = yes` a furniture piece stays walk-through strokes (no cells): the
/// default is unchanged for every existing level.
#[test]
fn furniture_trace_without_solid_stamps_no_cells() {
    for shape in SOLID_FURNITURE_SHAPES {
        let lvl = level(&prop(
            "furniture_trace",
            &format!("shape = {shape}\nat = 6.1 5.1 1.0\nsize = 1.5 1"),
        ));
        let exp = expand(&lvl);
        assert!(exp.cells.is_empty(), "{shape}");
        assert!(!exp.lines.is_empty(), "{shape} still draws its trace");
    }
}

/// `solid = yes` obeys the solid-kind rules: a shape without a solid form, or a bad
/// value, is a located parse error; a block on a puzzle cell is the keep-out error
/// naming both.
#[test]
fn solid_furniture_is_validated_and_kept_out_of_the_puzzle() {
    let e = load_err(&prop(
        "furniture_trace",
        "shape = rail\nat = 6.1 5.1 1.0\nsize = 3 0.2\nsolid = yes",
    ));
    assert!(e.msg.contains("solid") && e.msg.contains("rail"), "{e}");
    let e = load_err(&prop(
        "furniture_trace",
        "shape = bench\nat = 6.1 5.1 1.0\nsolid = maybe",
    ));
    assert!(e.msg.contains("solid"), "{e}");
    // The switch column at (6,20,*): a trace may stand there, a solid block may not.
    let trace = level(&prop(
        "furniture_trace",
        "shape = bench\nat = 6.1 20.1 1.0\nsize = 0.8 0.8",
    ));
    let (_, errs) = expand_level(&trace, &trace.build_map_base());
    assert!(errs.is_empty(), "{errs:?}");
    let e = load_err(&prop(
        "furniture_trace",
        "shape = bench\nat = 6.1 20.1 1.0\nsize = 0.8 0.8\nsolid = yes",
    ));
    assert!(
        e.msg.contains("[prop p]") && e.msg.contains("switch"),
        "{e}"
    );
}

#[test]
fn zband_suffix_format() {
    assert_eq!(zband_suffix(0.0, 0.5, None), "~z0_50");
    assert_eq!(
        zband_suffix(0.0, 0.42, Some((0xFFD080, 0.7))),
        "~z0_42~tffd080_70"
    );
}

// ─────────────── `[meta] prop_style` (two-repo split A1, replaces the
// `campaign.toml`-derived default) ───────────────────────────────────────────

/// `ROOM` with a `[meta] prop_style = <style>` key spliced in right after
/// `title`, so a test can pick the level's default pack without hand-writing a
/// whole fixture. `props` is appended exactly like [`level`].
fn level_with_prop_style(style: &str, props: &str) -> Level {
    let room = ROOM.replacen(
        "title = prop test\n\n[room]",
        &format!("title = prop test\nprop_style = {style}\n\n[room]"),
        1,
    );
    assert!(room != ROOM, "splice point not found in ROOM");
    let txt = format!("{room}{props}");
    load_string(&txt, "prop-test.lvl").unwrap_or_else(|e| panic!("level must load: {e}\n{props}"))
}

/// Absent `[meta] prop_style` ⇒ the struct default (`Lab`), and NOT authored —
/// so `re_voximport::to_lvl_text` omits the key and every level that predates
/// it stays byte-identical across a save (see `Level::prop_style_authored`).
#[test]
fn prop_style_absent_defaults_to_lab_unauthored() {
    let lvl = level("");
    assert_eq!(lvl.prop_default_style, PropStyle::Lab);
    assert!(
        !lvl.prop_style_authored,
        "an absent key must not be marked authored"
    );
}

/// Every named pack parses through `[meta] prop_style`, and is marked authored.
#[test]
fn prop_style_parses_every_named_pack() {
    for style in PropStyle::ALL {
        let lvl = level_with_prop_style(style.as_str(), "");
        assert_eq!(lvl.prop_default_style, style, "{}", style.as_str());
        assert!(
            lvl.prop_style_authored,
            "{}: must be marked authored",
            style.as_str()
        );
    }
}

/// An unrecognized `[meta] prop_style` value is a load error naming the line
/// and the key — the same discipline as `[material] surface` — because a typo
/// here would silently repaint every undressed prop in the level.
#[test]
fn prop_style_unknown_value_is_a_load_error() {
    let room = ROOM.replacen(
        "title = prop test\n\n[room]",
        "title = prop test\nprop_style = nonsense\n\n[room]",
        1,
    );
    match load_string(&room, "prop-test.lvl") {
        Ok(_) => panic!("expected a parse error for an unknown prop_style value"),
        Err(e) => assert!(e.msg.contains("prop_style"), "error must name the key: {e}"),
    }
}

/// The level's own `[meta] prop_style` — not a per-prop `style=` override — is
/// what an undressed prop resolves to. This is the mechanism's only remaining
/// test surface for "a prop names no style" now that the default no longer
/// comes from `campaign.toml`.
#[test]
fn level_prop_style_is_the_undressed_prop_default() {
    let undressed = "at = 11 10 1\nfacing = +y\nsize = 4 1 4";

    // ROOM carries no `[meta] prop_style` ⇒ the struct default (`lab`) applies.
    let lab_exp = expand(&level(&prop("shelving", undressed)));
    let lab_body = pack_material(PropStyle::Lab, MatSlot::Body);
    assert!(lab_exp
        .cells
        .iter()
        .any(|c| c.mat.as_deref() == Some(lab_body.name.as_str())));

    // A level that authors `prop_style = archive` resolves the SAME undressed
    // prop to the archive pack instead — proving the level-level default, not
    // just the per-prop `style=` override tested by
    // `material_pack_is_applied_and_overridable`, drives the pack choice.
    let archive_lvl = level_with_prop_style("archive", &prop("shelving", undressed));
    assert_eq!(archive_lvl.prop_default_style, PropStyle::Archive);
    let archive_exp = expand(&archive_lvl);
    let archive_body = pack_material(PropStyle::Archive, MatSlot::Body);
    assert!(archive_exp
        .cells
        .iter()
        .any(|c| c.mat.as_deref() == Some(archive_body.name.as_str())));
}
