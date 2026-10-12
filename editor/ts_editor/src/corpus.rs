//! THE CORPUS INVARIANT — every cycling field of every object of every SHIPPED
//! level, driven through the editor's real `c` key.
//!
//! ## Why this file exists
//!
//! `rowtype.rs` already carried `every_enum_option_list_contains_its_current_value`,
//! which is exactly the right assertion — but it ran against `placed(kind)`:
//! objects the editor had just created with its own DEFAULT values. A default is
//! by construction on the option list, so that test could never see a single
//! authored value and never failed while SEVEN live instances of the
//! DESTRUCTIVE-SNAP bug class sat in the crate (a node glyph snapping to `N`, a
//! material preset snapping to *none*, an emitter direction snapping to `+Z`, a
//! glyph set snapping to `none`, a material glyph snapping to `+`, a dangling
//! material binding snapping to `(none)`, a non-cardinal spawn facing snapping to
//! `+y`) — 223 authored values across the corpus, each one keypress from being
//! unrecoverable.
//!
//! This module points the same idea at `levels/*.lvl` — the real corpus, real
//! authored values — and adds the assertion the option-list check alone cannot
//! make: that pressing `c` a full lap RETURNS the field to where it started.
//! Between them they are a standing proof that no keypress in the editor can
//! silently replace an authored value with a list's first entry.
//!
//! ## What is asserted, per (level, object, field)
//!
//! 1. **Reachability** — if the field classifies as [`RowValue::Enum`], its
//!    option list contains its current value. A popup that cannot show the
//!    author what they have is a bug (`rowedit::open_for` highlights row 0 for
//!    anything it cannot find).
//! 2. **Lap identity** — pressing the field's own `c` key `options.len()` times
//!    leaves the readout byte-identical to where it started. This is the property
//!    the `unwrap_or(0)` snap broke: it made the FIRST press unrecoverable.
//!
//! Both run over every object of every kind, so growth in the corpus grows the
//! test for free.
//!
//! TEST-ONLY: the whole module is `#[cfg(test)]`, so it costs the shipped binary
//! nothing.

use crate::rowvalue::RowValue;
use crate::state::{EditorState, EmitterField, EntityField, EntityKind};

/// Point `state`'s selection registers at instance `i` of `kind`, and report
/// whether that instance exists. Singletons (`Door` / `Spawn` / `Pedestal`) have
/// exactly one instance and no index.
///
/// Exhaustive over [`EntityKind`] on purpose: a new kind added to the selector
/// stops compiling here until the corpus sweep learns how to select it.
fn select(state: &mut EditorState, kind: EntityKind, i: usize) -> bool {
    state.place = Some(kind);
    if i >= count(state, kind) {
        return false;
    }
    // SYNTHESIZED emitters (the `[effects] snow`/`rain` swarms `re_level` builds at
    // load) are not selectable in the editor at all — `entity::emitter_at` and
    // `tree.rs` both filter them out, and no writer serializes them. Including them
    // would inflate every count here with objects no keypress can reach.
    let i = if kind == EntityKind::Emitter {
        authored_emitters(state)[i]
    } else {
        i
    };
    let idx = Some(i);
    match kind {
        EntityKind::Door | EntityKind::Spawn | EntityKind::Pedestal => {}
        EntityKind::Part => state.selected_part = idx,
        EntityKind::Node => state.selected_node = idx,
        EntityKind::Emitter => state.selected_emitter = idx,
        EntityKind::Trigger => state.selected_trigger = idx,
        EntityKind::Pipe => state.selected_pipe = idx,
        EntityKind::ZoneInput => state.selected_zone_input = idx,
        EntityKind::Note => state.selected_note = idx,
        EntityKind::WallDecal => state.selected_decal = idx,
        EntityKind::Holo => state.selected_holo = idx,
        EntityKind::Portal => state.selected_portal = idx,
        EntityKind::Zone => state.selected_zone = idx,
        EntityKind::Line => state.selected_line = idx,
        EntityKind::Inset => state.selected_inset = idx,
        EntityKind::Field => state.selected_field = idx,
    }
    true
}

/// How many instances of `kind` the level holds (1 for the singletons).
fn count(state: &EditorState, kind: EntityKind) -> usize {
    let lvl = &state.level;
    match kind {
        EntityKind::Door | EntityKind::Spawn | EntityKind::Pedestal => 1,
        EntityKind::Part => lvl.parts.len(),
        EntityKind::Node => lvl.nodes.len(),
        EntityKind::Emitter => authored_emitters(state).len(),
        EntityKind::Trigger => lvl.triggers.len(),
        EntityKind::Pipe => lvl.pipes.len(),
        EntityKind::ZoneInput => lvl.zone_inputs.len(),
        EntityKind::Note => lvl.notes.len(),
        EntityKind::WallDecal => lvl.decals.len(),
        EntityKind::Holo => lvl.holos.len(),
        EntityKind::Portal => lvl.portals.len(),
        EntityKind::Zone => lvl.overlay.zones.len(),
        EntityKind::Line => lvl.overlay.lines.len(),
        EntityKind::Inset => lvl.overlay.insets.len(),
        EntityKind::Field => lvl.field_defs.len(),
    }
}

/// Indices of the emitters the editor lets you SELECT — the authored ones.
/// Mirrors `entity::emitter_at` / `tree.rs`'s `!e.synthesized` filter.
fn authored_emitters(state: &EditorState) -> Vec<usize> {
    state
        .level
        .emitters
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.synthesized)
        .map(|(i, _)| i)
        .collect()
}

/// The 17 [`EntityKind`]s via the object-selector's own `next()` cycle, so this
/// list can never drift from the selector's coverage.
fn all_kinds() -> Vec<EntityKind> {
    let mut k = EntityKind::Door;
    let mut out = Vec::new();
    for _ in 0..17 {
        out.push(k);
        k = k.next();
    }
    out
}

/// One row under test: how to read it, and how to press its `c` key.
enum Row {
    Entity(EntityField),
    Emitter(EmitterField),
    Material(crate::material::MaterialField),
}

impl Row {
    /// The row's typed value, exactly as the inspector and the popup see it.
    fn value(&self, state: &EditorState) -> RowValue {
        match *self {
            Self::Entity(f) => crate::rowtype::entity_value(state, f),
            Self::Emitter(f) => crate::rowtype::emitter_value(state, f),
            Self::Material(f) => crate::rowtype::material_value(state, f),
        }
    }

    /// Press the row's own `c` key ONCE, through the real dispatch.
    fn cycle(&self, state: &mut EditorState) {
        match *self {
            Self::Entity(f) => {
                state.entity_field = f;
                crate::entity::cycle_property(state);
            }
            Self::Emitter(f) => {
                state.emitter_field = f;
                crate::entity::cycle_property(state);
            }
            // Material rows split across TWO keys: `c` for the declared enum
            // fields, `+` for the candidate-set fields (glyph, recipe) that have
            // always stepped with `+`/`-`. Pressing the wrong one is a hinted
            // no-op, which would make this sweep silently vacuous for those rows —
            // exactly the failure mode this module exists to end.
            Self::Material(f) => {
                state.material_field = f;
                if f.is_enum() {
                    crate::material::cycle_enum(state, 1);
                } else {
                    crate::material::adjust_field(state, 1);
                }
            }
        }
    }

    fn name(&self) -> String {
        match *self {
            Self::Entity(f) => format!("{f:?}"),
            Self::Emitter(f) => format!("emitter.{f:?}"),
            Self::Material(f) => format!("material.{f:?}"),
        }
    }
}

/// Run `check` over every (object, row) pair of the loaded `state`, collecting
/// human-readable failures. `where_` names the level for the report.
fn for_every_row(
    state: &mut EditorState,
    where_: &str,
    failures: &mut Vec<String>,
    mut check: impl FnMut(&mut EditorState, &Row, &str, &mut Vec<String>),
) {
    for kind in all_kinds() {
        for i in 0..count(state, kind) {
            if !select(state, kind, i) {
                continue;
            }
            let rows: Vec<Row> = if kind == EntityKind::Emitter {
                crate::state::EMITTER_FIELDS
                    .iter()
                    .map(|&f| Row::Emitter(f))
                    .collect()
            } else {
                crate::state::entity_fields_for(kind)
                    .iter()
                    .map(|&f| Row::Entity(f))
                    .collect()
            };
            for row in &rows {
                let at = format!("{where_} {kind:?}#{i} {}", row.name());
                check(state, row, &at, failures);
            }
        }
    }
    // Materials are selected by `material_sel`, not by an `EntityKind`.
    let unique = &crate::material::MATERIAL_FIELDS[..crate::material::MATERIAL_FIELDS.len() - 1];
    for i in 0..state.level.materials.len() {
        state.material_sel = i;
        state.material_mode = true;
        for &f in unique {
            let row = Row::Material(f);
            let name = state.level.materials[i].name.clone();
            let at = format!("{where_} material '{name}' {}", row.name());
            check(state, &row, &at, failures);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::shipped_levels;

    /// Open one shipped level exactly as the editor does (`open::load_path`),
    /// through a private scratch copy so nothing in `levels/` is touched.
    fn opened(src: &std::path::Path, name: &str) -> Option<EditorState> {
        let text = std::fs::read_to_string(src).unwrap_or_else(|e| panic!("read {name}: {e}"));
        let file = crate::testutil::ScratchFile::write(
            crate::testutil::scratch_path(&format!("corpus-cycle-{name}")),
            &text,
        );
        let mut s = EditorState::new_empty(4, 4, 2);
        if crate::open::load_path(&mut s, file.path()) {
            Some(s)
        } else {
            None
        }
    }

    /// INVARIANT 1 — every `Enum` row of every object of every shipped level has
    /// its CURRENT value in its own option list.
    ///
    /// This is the assertion `rowtype`'s version makes against freshly-placed
    /// defaults; pointed at real authored data it is what catches a hand-written
    /// option list that has fallen behind the format.
    #[test]
    fn every_enum_option_list_contains_its_current_value_across_the_shipped_corpus() {
        let mut failures: Vec<String> = Vec::new();
        let mut checked = 0usize;
        for (name, src) in shipped_levels() {
            let Some(mut s) = opened(&src, &name) else {
                failures.push(format!("  {name}: the editor could not open it"));
                continue;
            };
            for_every_row(&mut s, &name, &mut failures, |st, row, at, fails| {
                if let RowValue::Enum { current, options } = row.value(st) {
                    if !options.contains(&current) {
                        fails.push(format!(
                            "  {at}: current {current:?} is NOT in its own options {options:?}"
                        ));
                    }
                }
            });
            checked += 1;
        }
        assert!(
            checked >= 40,
            "expected the whole corpus, only opened {checked} levels"
        );
        assert!(
            failures.is_empty(),
            "{} enum row(s) cannot show their own value:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// INVARIANT 2 — a FULL LAP of `c` is the identity, for every cycling row of
    /// every object of every shipped level.
    ///
    /// The destructive snap is invisible to invariant 1 the moment an option list
    /// is built from the live value; this one is what proves the value can still
    /// be REACHED after the first press. Pressing `options.len()` times must land
    /// exactly back on the authored readout.
    #[test]
    fn a_full_lap_of_the_cycle_key_restores_every_authored_value_in_the_corpus() {
        let mut failures: Vec<String> = Vec::new();
        for (name, src) in shipped_levels() {
            let Some(mut s) = opened(&src, &name) else {
                continue; // invariant 1 already reports an unopenable level
            };
            for_every_row(&mut s, &name, &mut failures, |st, row, at, fails| {
                let RowValue::Enum { current, options } = row.value(st) else {
                    return;
                };
                for _ in 0..options.len() {
                    row.cycle(st);
                }
                let after = row.value(st).display();
                if after != current {
                    fails.push(format!(
                        "  {at}: {} presses of `c` turned {current:?} into {after:?} \
                         (options {options:?}) — the authored value is unreachable",
                        options.len()
                    ));
                }
            });
        }
        assert!(
            failures.is_empty(),
            "{} cycling row(s) do not close their lap:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Opening a level SEEDS the vocabulary from that file — so the authored
    /// glyphs are one popup away, not one full lap away.
    #[test]
    fn opening_a_level_seeds_the_vocabulary_from_what_it_authors() {
        let s = opened(std::path::Path::new("../../levels/01.lvl"), "01.lvl").expect("open");
        assert!(
            s.vocab.node_glyph.contains(&'R'),
            "01.lvl's authored `glyph = R` should be offered immediately, got {:?}",
            s.vocab.node_glyph
        );
        // And the seed is exactly the off-canonical set: nothing NODE_GLYPHS
        // already offers is duplicated into it.
        for g in &s.vocab.node_glyph {
            assert!(
                !crate::entity::NODE_GLYPHS.contains(g),
                "{g:?} is already canonical and must not be duplicated into the extras"
            );
        }
    }

    /// And the NEXT level's vocabulary is its own — the previous document's
    /// authored values do not leak into it.
    #[test]
    fn opening_a_second_level_does_not_inherit_the_firsts_vocabulary() {
        let text = std::fs::read_to_string("../../levels/01.lvl").expect("read 01.lvl");
        let first = crate::testutil::ScratchFile::write(
            crate::testutil::scratch_path("vocab-leak-a"),
            &text,
        );
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(crate::open::load_path(&mut s, first.path()));
        assert!(s.vocab.node_glyph.contains(&'R'));

        let text2 = std::fs::read_to_string("../../levels/02.lvl").expect("read 02.lvl");
        let second = crate::testutil::ScratchFile::write(
            crate::testutil::scratch_path("vocab-leak-b"),
            &text2,
        );
        assert!(crate::open::load_path(&mut s, second.path()));
        let mut expected = crate::vocab::Vocab::default();
        expected.harvest(&s.level);
        assert_eq!(
            s.vocab, expected,
            "the second document's vocabulary must be exactly its own"
        );
    }

    /// Harvesting twice changes nothing (the seed is a SET, not a log).
    #[test]
    fn harvesting_is_idempotent() {
        let s = opened(std::path::Path::new("../../levels/01.lvl"), "01.lvl").expect("open");
        let once = s.vocab.clone();
        let mut twice = once.clone();
        twice.harvest(&s.level);
        assert_eq!(once, twice);
    }

    // ───────────────── NAMED REGRESSIONS — one per confirmed instance ─────────────────
    //
    // Each opens a REAL shipped level, selects the object that authors the value,
    // and asserts the two properties the destructive snap broke:
    //
    //   (a) after ONE `c` press the authored value is still an OPTION — i.e. the
    //       author can get back to it. Under the old code it vanished from the
    //       world on that press.
    //   (b) a FULL LAP of `c` lands exactly back on it.
    //
    // (a) is the sharp one: `c` legitimately changes the value, so "it changed"
    // is not the bug — "it can never come back" is.

    /// Read `row`'s value, press `c` once, and assert the ORIGINAL is still an
    /// option; then finish the lap and assert it comes back exactly.
    #[track_caller]
    fn assert_survives_a_press(state: &mut EditorState, row: &Row, what: &str) {
        let RowValue::Enum { current, options } = row.value(state) else {
            panic!("{what}: expected an Enum row, got {:?}", row.value(state));
        };
        assert!(
            options.contains(&current),
            "{what}: {current:?} not in {options:?}"
        );
        let lap = options.len();
        row.cycle(state);
        let after_one = row.value(state);
        let RowValue::Enum { options: opts2, .. } = &after_one else {
            panic!("{what}: row stopped being an Enum after one press");
        };
        assert!(
            opts2.contains(&current),
            "{what}: ONE `c` press made the authored value {current:?} unreachable \
             (options are now {opts2:?})"
        );
        for _ in 1..lap {
            row.cycle(state);
        }
        assert_eq!(
            row.value(state).display(),
            current,
            "{what}: {lap} presses of `c` did not return to the authored value"
        );
    }

    /// B1 — `levels/01.lvl` authors `[node R] glyph = R`. `R` is not on
    /// `NODE_GLYPHS`, so one `c` used to replace it with `N` forever. The glyph is
    /// load-bearing: `re_game/src/story.rs` uses it as the node's display token.
    #[test]
    fn b1_the_authored_node_glyph_in_01_lvl_survives_the_cycle_key() {
        let mut s = opened(std::path::Path::new("../../levels/01.lvl"), "01.lvl").expect("open");
        let i = s
            .level
            .nodes
            .iter()
            .position(|n| n.glyph == 'R')
            .expect("01.lvl authors a node with glyph R");
        assert!(select(&mut s, EntityKind::Node, i));
        assert_survives_a_press(
            &mut s,
            &Row::Entity(EntityField::NodeGlyph),
            "01.lvl node R glyph",
        );
    }

    /// B2a — WIDENING. `preset = void` is a real `re_render` catalog material that
    /// the editor's hand-typed 6-name list did not offer; 14 instances of it were
    /// destroyed by one `c`. It is now canonical because the list IS the catalog.
    #[test]
    fn b2_the_authored_void_preset_is_now_a_catalog_option() {
        let mut s = opened(
            std::path::Path::new("../../levels/20-stairs-straight.lvl"),
            "20-stairs-straight.lvl",
        )
        .expect("open");
        let i = s
            .level
            .materials
            .iter()
            .position(|m| m.preset.as_deref() == Some("void"))
            .expect("20-stairs-straight.lvl authors preset = void");
        s.material_sel = i;
        s.material_mode = true;
        assert!(
            crate::vocab::preset_catalog().contains(&Some("void".to_string())),
            "`void` is a renderer catalog name and must be offered outright"
        );
        assert_survives_a_press(
            &mut s,
            &Row::Material(crate::material::MaterialField::Preset),
            "20-stairs-straight.lvl preset void",
        );
    }

    /// B2b — PRESERVING. `facet` is a real demo material the catalog carries no
    /// NAME entry for, so widening alone cannot save it; it survives because an
    /// off-list value is remembered rather than snapped.
    #[test]
    fn b2_the_authored_facet_preset_survives_even_though_the_catalog_has_no_name_for_it() {
        let mut s = opened(
            std::path::Path::new("../../levels/95-facet-demo.lvl"),
            "95-facet-demo.lvl",
        )
        .expect("open");
        let i = s
            .level
            .materials
            .iter()
            .position(|m| m.preset.as_deref() == Some("facet"))
            .expect("95-facet-demo.lvl authors preset = facet");
        s.material_sel = i;
        s.material_mode = true;
        assert!(
            !crate::vocab::preset_catalog().contains(&Some("facet".to_string())),
            "premise: the renderer catalog has no NAME entry for `facet`"
        );
        assert_survives_a_press(
            &mut s,
            &Row::Material(crate::material::MaterialField::Preset),
            "95-facet-demo.lvl preset facet",
        );
    }

    /// `levels/21-emitter-showcase.lvl` authors `direction = 1.0 0.0 0.25`.
    /// One `c` used to rotate that emitter ~76° to `+Z` with no way back.
    #[test]
    fn m5_the_authored_emitter_direction_survives_the_cycle_key() {
        let mut s = opened(
            std::path::Path::new("../../levels/21-emitter-showcase.lvl"),
            "21-emitter-showcase.lvl",
        )
        .expect("open");
        let i = s
            .level
            .emitters
            .iter()
            .position(|e| (e.direction.z - 0.25).abs() < 1e-3 && (e.direction.x - 1.0).abs() < 1e-3)
            .expect("21-emitter-showcase.lvl authors direction = 1.0 0.0 0.25");
        assert!(select(&mut s, EntityKind::Emitter, i));
        assert_survives_a_press(
            &mut s,
            &Row::Emitter(EmitterField::DirectionAxis),
            "21-emitter-showcase.lvl custom direction",
        );
    }

    /// The same level authors `glyph_set = · : * ✦ ✧`, which is not one of
    /// the five `GLYPH_SETS` presets. One `c` used to erase it to `none`.
    #[test]
    fn the_authored_emitter_glyph_set_survives_the_cycle_key() {
        let mut s = opened(
            std::path::Path::new("../../levels/21-emitter-showcase.lvl"),
            "21-emitter-showcase.lvl",
        )
        .expect("open");
        let i = s
            .level
            .emitters
            .iter()
            .position(|e| e.glyph_set.is_some())
            .expect("21-emitter-showcase.lvl authors an explicit glyph_set");
        assert!(select(&mut s, EntityKind::Emitter, i));
        assert_survives_a_press(
            &mut s,
            &Row::Emitter(EmitterField::GlyphSet),
            "21-emitter-showcase.lvl explicit glyph_set",
        );
    }

    /// The reported extra — `levels/12-mirror.lvl` authors `[material] glyph = #`,
    /// which is deliberately NOT on the candidate list. One `+`/`-` used to erase
    /// it. (`re_level`'s lone-`#` exception makes that value round-trip, so it is
    /// real authored data, not a parse artefact.)
    #[test]
    fn the_authored_material_glyph_hash_survives_the_step_key() {
        let mut s = opened(
            std::path::Path::new("../../levels/12-mirror.lvl"),
            "12-mirror.lvl",
        )
        .expect("open");
        let i = s
            .level
            .materials
            .iter()
            .position(|m| m.glyph.and_then(char::from_u32) == Some('#'))
            .expect("12-mirror.lvl authors [material] glyph = #");
        s.material_sel = i;
        s.material_mode = true;
        s.material_field = crate::material::MaterialField::Glyph;
        let RowValue::Enum { current, options } =
            crate::rowtype::material_value(&s, crate::material::MaterialField::Glyph)
        else {
            panic!("material glyph should classify as an Enum row");
        };
        assert_eq!(current, "#");
        assert!(options.contains(&current), "options {options:?} missing #");
        // Glyph is stepped by `+`/`-` (not `c`), so drive its real stepper.
        let lap = options.len();
        crate::material::adjust_field(&mut s, 1);
        assert_ne!(
            s.level.materials[i].glyph.and_then(char::from_u32),
            Some('#'),
            "one press must actually move"
        );
        for _ in 1..lap {
            crate::material::adjust_field(&mut s, 1);
        }
        assert_eq!(
            s.level.materials[i].glyph.and_then(char::from_u32),
            Some('#'),
            "a full lap of `+` must return the authored `#` glyph"
        );
    }

    /// GODRAYS-demo authors `[material] glyph = ^` — the second confirmed
    /// material-glyph victim.
    #[test]
    fn the_authored_material_glyph_caret_is_reachable() {
        let mut s = opened(
            std::path::Path::new("../../levels/GODRAYS-demo.lvl"),
            "GODRAYS-demo.lvl",
        )
        .expect("open");
        let i = s
            .level
            .materials
            .iter()
            .position(|m| m.glyph.and_then(char::from_u32) == Some('^'))
            .expect("GODRAYS-demo.lvl authors [material] glyph = ^");
        s.material_sel = i;
        s.material_mode = true;
        let RowValue::Enum { current, options } =
            crate::rowtype::material_value(&s, crate::material::MaterialField::Glyph)
        else {
            panic!("expected Enum");
        };
        assert_eq!(current, "^");
        assert!(options.contains(&current), "options {options:?} missing ^");
    }

    /// LATENT, and fixed anyway — `re_level` accepts FREE TEXT after
    /// `action = event`, so a level may carry a label the 6-entry preset list does
    /// not contain. No shipped level does today; "latent" was the wrong call on the
    /// decal bug, so it is closed here rather than noted.
    #[test]
    fn a_free_text_trigger_event_label_survives_the_cycle_key() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Trigger);
        s.cursor = (2, 2);
        s.z = 0;
        crate::entity::place_at_cursor(&mut s);
        s.level.triggers[0].action = re_core::TriggerAction::event("choir_swells");
        s.selected_trigger = Some(0);
        s.entity_field = EntityField::TrigTarget;
        let lap = crate::state::event_label_canon().len() + 1;
        crate::entity::cycle_property(&mut s);
        let after_one = label_of(&s);
        assert_ne!(after_one, "choir_swells", "one press must actually move");
        for _ in 1..lap {
            crate::entity::cycle_property(&mut s);
        }
        assert_eq!(
            label_of(&s),
            "choir_swells",
            "a full lap of `c` must return the authored free-text event label"
        );
    }

    /// The selected trigger's event label.
    fn label_of(state: &EditorState) -> String {
        match &state.level.triggers[0].action {
            re_core::TriggerAction::Event { label } => label.clone(),
            other => panic!("expected an Event action, got {other:?}"),
        }
    }

    /// LATENT, and fixed anyway — `re_level::facing_to_yaw_col` falls through to
    /// `other.parse()`, so `facing = 47` is a legal authored spawn yaw. `f`/`c`
    /// used to snap it to `+y`.
    #[test]
    fn a_non_cardinal_spawn_facing_survives_the_cycle_key() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.level.player.facing_yaw_col = 47;
        s.place = Some(EntityKind::Spawn);
        s.entity_field = EntityField::SpawnFacing;
        let lap = crate::state::facing_canon().len() + 1;
        crate::entity::cycle_property(&mut s);
        assert_ne!(
            s.level.player.facing_yaw_col, 47,
            "one press must actually move"
        );
        for _ in 1..lap {
            crate::entity::cycle_property(&mut s);
        }
        assert_eq!(
            s.level.player.facing_yaw_col, 47,
            "a full lap of `c` must return the authored non-cardinal yaw"
        );
    }

    /// LATENT, and fixed anyway — a zone/line/portal may name a material the level
    /// does not define (an import, a rename in flight, a forward reference).
    /// Cycling one used to snap the binding to `(none)`, deleting it outright.
    #[test]
    fn a_dangling_material_binding_survives_the_cycle_key() {
        let mut s = EditorState::new_empty(8, 8, 4);
        crate::material::new_material(&mut s); // one DEFINED material to cycle past
        s.place = Some(EntityKind::Zone);
        s.cursor = (2, 2);
        s.z = 0;
        crate::entity::place_at_cursor(&mut s);
        s.level.overlay.zones[0].material = re_core::MaterialId("not_in_this_level".to_string());
        s.selected_zone = Some(0);
        s.entity_field = EntityField::ZoneMaterial;
        let row = Row::Entity(EntityField::ZoneMaterial);
        assert_survives_a_press(&mut s, &row, "dangling zone material binding");
        assert_eq!(s.level.overlay.zones[0].material.0, "not_in_this_level");
    }

    /// AN INSTANCE THE SWEEP DID NOT NAME — the recipe cycle discarded TUNED
    /// PARAMETERS, not just off-list values.
    ///
    /// `levels/12-mirror.lvl` authors `recipe = grain scale=2.5 speck=0.6
    /// tile_eps=0.06`. The cycle stepped through `RECIPES`, whose entries carry
    /// each variant's DEFAULT parameters, so ONE `+` press replaced the tuned
    /// grain with the next variant's defaults and stepping back returned a
    /// *different* grain — the label matched, the data did not. That is the same
    /// bug class one layer down: a fixed list cannot represent authored data, so
    /// the cycle silently substituted what it could represent.
    #[test]
    fn a_tuned_recipe_survives_a_full_lap_of_the_step_key() {
        let mut s = opened(
            std::path::Path::new("../../levels/12-mirror.lvl"),
            "12-mirror.lvl",
        )
        .expect("open");
        let i = s
            .level
            .materials
            .iter()
            .position(|m| matches!(m.recipe, Some(re_world::MaterialRecipe::Grain { .. })))
            .expect("12-mirror.lvl authors a grain recipe");
        let authored = s.level.materials[i].recipe.expect("grain");
        let re_world::MaterialRecipe::Grain { scale, .. } = authored else {
            panic!("expected Grain");
        };
        assert!(
            (scale - 2.5).abs() < 1e-6,
            "premise: the authored grain is tuned (scale=2.5)"
        );

        s.material_sel = i;
        s.material_mode = true;
        s.material_field = crate::material::MaterialField::Recipe;
        let lap = crate::material::recipe_variant_options(&s).len();
        for _ in 0..lap {
            crate::material::adjust_field(&mut s, 1);
        }
        assert_eq!(
            s.level.materials[i].recipe,
            Some(authored),
            "a full lap of `+` must return the AUTHORED grain, parameters and all"
        );
    }

    /// The family list is now the renderer's own library, so the three `state_*`
    /// families that the hand-copied 13-name list omitted are authorable.
    #[test]
    fn the_state_glyph_families_are_reachable_from_the_editor() {
        let fams = crate::vocab::emitter_families();
        for want in ["state_node", "state_door", "state_trace"] {
            assert!(
                fams.contains(&want.to_string()),
                "family {want} is not offered"
            );
        }
    }

    /// The corpus sweep is only as good as its reach: assert it actually visits a
    /// substantial number of (object, row) pairs, so a future refactor that
    /// quietly stops selecting anything cannot leave two green vacuous tests
    /// behind. (This is precisely how `rowtype`'s `placed(kind)` version became
    /// vacuous.)
    #[test]
    fn the_corpus_sweep_actually_visits_the_corpus() {
        let mut visited = 0usize;
        let mut enum_rows = 0usize;
        let mut off_canon = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for (name, src) in shipped_levels() {
            let Some(mut s) = opened(&src, &name) else {
                continue;
            };
            // The extras harvested at load ARE the authored values the old
            // steppers destroyed; count them so a regression in `harvest` shows
            // up as a number, not as silence.
            off_canon += s.vocab.node_glyph.len()
                + s.vocab.preset.len()
                + s.vocab.material_glyph.len()
                + s.vocab.direction.len()
                + s.vocab.glyph_set.len()
                + s.vocab.family.len()
                + s.vocab.event_label.len()
                + s.vocab.facing.len()
                + s.vocab.material_name.len();
            for_every_row(&mut s, &name, &mut failures, |st, row, _at, _f| {
                visited += 1;
                if matches!(row.value(st), RowValue::Enum { .. }) {
                    enum_rows += 1;
                }
            });
        }
        assert!(
            visited > 5_000,
            "only {visited} (object, row) pairs visited"
        );
        assert!(enum_rows > 500, "only {enum_rows} enum rows visited");
        assert!(
            off_canon > 100,
            "only {off_canon} off-canonical authored values found in the whole corpus — \
             the harvest is not seeing the levels"
        );
    }
}
