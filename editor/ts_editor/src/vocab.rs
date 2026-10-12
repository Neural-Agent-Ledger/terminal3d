//! THE ONE RULE for every `c`-cycle over a fixed option list: **a value that is
//! not on the list is PRESERVED and participates in the cycle — never erased.**
//!
//! ## The bug class this module exists to make impossible
//!
//! Every cycling field used to be written the same way:
//!
//! ```ignore
//! let i = LIST.iter().position(|x| x == current).unwrap_or(0);
//! LIST[(i + delta).rem_euclid(LIST.len())]
//! ```
//!
//! The `unwrap_or(0)` (equivalently `None => LIST[0]`) is a **destructive snap**:
//! for any value the hand-written `LIST` does not happen to contain, ONE keypress
//! replaces authored data with the list's first entry, and NO key can ever get
//! the original back. It shipped repeatedly before this module existed —
//! `step_palette` snapped every off-palette colour to white and `DECAL_SURFACES`
//! snapped imported decals to `trace` (both fixed earlier); the sweep that
//! produced this module then found it live in seven more places. Measured against
//! the shipped corpus by `corpus::tests`, those seven together put **223 authored
//! values across `levels/*.lvl` one keypress away from being unrecoverable**:
//! 149 node glyphs, 65 material presets, 6 material glyphs, 1 emitter direction,
//! 1 emitter glyph set and 1 portal skin. It is now zero.
//!
//! ## The rule
//!
//! A cycling field has a CANONICAL list (the vocabulary the editor offers) and,
//! per session, a set of EXTRAS — off-list values this editor has seen. The list
//! a `c` press walks is
//!
//! ```text
//! canonical ++ extras ++ [current, when neither already holds it]
//! ```
//!
//! and stepping is a plain rotation within it. Two properties follow, and both
//! are asserted by tests:
//!
//! * **Reachability** — the option list ALWAYS contains the current value, so the
//!   enum popup can never highlight the wrong row (`rowedit::open_for`) and the
//!   `c` key can never land somewhere the author cannot see.
//! * **Lap identity** — pressing `c` `options.len()` times returns the field to
//!   exactly where it started. This is the property the old code broke, and it is
//!   why the extras set is STICKY: a cycle first REMEMBERS the value it is
//!   stepping off (`remember_*`), so that value stays in the vocabulary for the
//!   rest of the session and the lap closes. Splicing only the live `current`
//!   would move the loss one keypress later instead of removing it.
//!
//! [`Vocab::harvest`] additionally seeds the extras from a level the moment it is
//! opened, so every authored glyph / preset / direction / family in the file is
//! offered from the first keypress rather than only after someone has stepped
//! off it.
//!
//! ## Widening vs. preserving
//!
//! Preserving is the floor; where the real domain is ENUMERABLE the canonical
//! list is widened to it instead of being hand-copied:
//!
//! * [`preset_catalog`] enumerates `re_render`'s own
//!   `ascii_shader::ids::CATALOG_NAMES` — 42 material names — rather than the 6
//!   hand-typed ones the editor used to offer, one of which (`concrete`) named a
//!   material that does not exist anywhere in the renderer.
//! * [`emitter_families`] enumerates `re_render::glyph_family::LIBRARY` rather
//!   than the 13 hand-typed names the editor used to offer (the library has 16).
//!
//! The other domains — a node glyph, a material glyph, an emission direction, a
//! glyph set, a trigger event label — are OPEN (any char / any unit vector / any
//! token), so there is no wider list to enumerate; for those, preserving IS the
//! whole fix.

use re_level::Level;

// ───────────────────────────── the generic rule ─────────────────────────────

/// The option list a cycle walks, plus the index `current` sits at within it.
///
/// Returning the index (rather than letting the caller search for it) is what
/// makes the `unwrap_or(0)` that caused this whole bug class *unwritable* at the
/// call sites: there is no lookup to get wrong.
///
/// `eq` is the field's own equality — a plain `==` for tokens, an epsilon
/// comparison for the float direction vector (where `==` is a deny-level clippy
/// finding and wrong besides).
pub fn options_at_by<T: Clone>(
    canon: &[T],
    extras: &[T],
    current: &T,
    eq: impl Fn(&T, &T) -> bool,
) -> (Vec<T>, usize) {
    let mut out: Vec<T> = canon.to_vec();
    for e in extras {
        if !out.iter().any(|o| eq(o, e)) {
            out.push(e.clone());
        }
    }
    if let Some(i) = out.iter().position(|o| eq(o, current)) {
        return (out, i);
    }
    out.push(current.clone());
    let i = out.len() - 1;
    (out, i)
}

/// [`options_at_by`] with `PartialEq`.
pub fn options_at<T: Clone + PartialEq>(canon: &[T], extras: &[T], current: &T) -> (Vec<T>, usize) {
    options_at_by(canon, extras, current, |a, b| a == b)
}

/// Rotate `delta` places from `current` through `options_at_by(..)`.
///
/// Never snaps, never panics, and — because the list always contains `current` —
/// `delta = options.len()` is the identity.
pub fn step_by<T: Clone>(
    canon: &[T],
    extras: &[T],
    current: &T,
    delta: i32,
    eq: impl Fn(&T, &T) -> bool,
) -> T {
    let (opts, i) = options_at_by(canon, extras, current, eq);
    if opts.is_empty() {
        return current.clone();
    }
    let n = opts.len() as i32;
    opts[(i as i32 + delta).rem_euclid(n) as usize].clone()
}

/// [`step_by`] with `PartialEq`.
pub fn step<T: Clone + PartialEq>(canon: &[T], extras: &[T], current: &T, delta: i32) -> T {
    step_by(canon, extras, current, delta, |a, b| a == b)
}

/// Record `current` in `extras` when the canonical list does not already offer
/// it — the STICKY half of the rule, and the reason a full lap of `c` returns to
/// where it started. Idempotent.
pub fn remember_by<T: Clone>(
    canon: &[T],
    extras: &mut Vec<T>,
    current: &T,
    eq: impl Fn(&T, &T) -> bool,
) {
    if canon.iter().any(|c| eq(c, current)) || extras.iter().any(|e| eq(e, current)) {
        return;
    }
    extras.push(current.clone());
}

/// [`remember_by`] with `PartialEq`.
pub fn remember<T: Clone + PartialEq>(canon: &[T], extras: &mut Vec<T>, current: &T) {
    remember_by(canon, extras, current, |a, b| a == b);
}

/// Equality for an emission direction: the same 1e-3 componentwise tolerance the
/// emitter cycle has always used to recognise a cardinal preset. (Exact float
/// `==` is a deny-level clippy finding here, and would call `+Z` "custom" for a
/// direction that merely round-tripped through the `.lvl` text.)
#[must_use]
pub fn dir_eq(a: &[f32; 3], b: &[f32; 3]) -> bool {
    (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3 && (a[2] - b[2]).abs() < 1e-3
}

// ──────────────────────────── enumerable domains ────────────────────────────

/// The renderer's OWN material catalog names, enumerated from
/// `re_render::ascii_shader::ids::CATALOG_NAMES`.
///
/// Hand-copying this list is what produced the bug: the editor offered 6 names
/// while shipped levels author 21, and one of the 6 (`concrete`) exists nowhere
/// in the renderer, so choosing it wrote a `.lvl` naming a material that silently
/// falls back to the surface base (`re_render/src/lib.rs`'s unknown-preset rule).
/// Enumerating means the editor can never again drift from the catalog.
#[must_use]
pub fn preset_catalog() -> Vec<Option<String>> {
    std::iter::once(None)
        .chain(
            re_render::ascii_shader::ids::CATALOG_NAMES
                .iter()
                .map(|n| Some((*n).to_string())),
        )
        .collect()
}

/// The glyph-family names, enumerated from `re_render::glyph_family::LIBRARY`.
///
/// Same drift, smaller blast radius: the editor hand-listed 13 of the library's
/// 16 families, so `state_node` / `state_door` / `state_trace` were unauthorable
/// and a level using one would have been snapped to `haze_o`.
#[must_use]
pub fn emitter_families() -> Vec<String> {
    re_render::glyph_family::LIBRARY
        .iter()
        .map(|f| f.name.to_string())
        .collect()
}

// ────────────────────────────── the session set ──────────────────────────────

/// Off-list values this editing session has seen, per cycling field.
///
/// Session-scoped on purpose: it is a VOCABULARY, not content. Undo/redo restore
/// `(Map, Level)` snapshots (`history::Snapshot`) and deliberately do not touch
/// this — an option that stays offered after an undo is harmless, whereas an
/// option that disappeared would break lap identity.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Vocab {
    /// Off-list `[node] glyph` values (the corpus authors 38 distinct glyphs; the
    /// canonical list offers 10).
    pub node_glyph: Vec<char>,
    /// Off-list `[material] glyph` values (`None` = "no fixed glyph").
    pub material_glyph: Vec<Option<char>>,
    /// Off-catalog `[material] preset` names.
    pub preset: Vec<Option<String>>,
    /// Off-library `[emitter] family` names.
    pub family: Vec<String>,
    /// Non-preset `[emitter] glyph_set` sequences.
    pub glyph_set: Vec<Option<Vec<char>>>,
    /// Non-cardinal `[emitter] direction` vectors.
    pub direction: Vec<[f32; 3]>,
    /// Free-text trigger `action = event LABEL` labels.
    pub event_label: Vec<String>,
    /// Non-cardinal player `facing` yaw values.
    pub facing: Vec<i32>,
    /// Material names referenced by a zone / line / portal that the level's
    /// `[material]` sections do not define (a dangling skin binding).
    pub material_name: Vec<String>,
    /// TUNED `[material] recipe` instances, at most one per variant. The recipe
    /// cycle steps through VARIANTS, and `RECIPES` carries only each variant's
    /// DEFAULT parameters — so without this, stepping off
    /// `grain scale=2.5 speck=0.6` and back handed you a different grain. See
    /// `material::step_recipe`.
    pub recipe: Vec<re_world::MaterialRecipe>,
    /// Non-cardinal `[field] facing` vectors (`crate::field::facing_to_str`'s
    /// literal `x,y,z` form for anything that isn't one of the six axes).
    pub field_facing: Vec<String>,
    /// Off-picker `[field] bind` values — a hand-authored `budget:K`/`seq:K`/
    /// `read:I` target (`crate::field::bind_options`' doc: the picker only
    /// offers node/pipe/door names, not these free-form targets).
    pub field_bind: Vec<String>,
    /// Off-picker `[field] drive` values — chiefly a hand-authored `ring:...`
    /// multi-pipe list (`crate::field::drive_options`'s doc: same scope cut).
    pub field_drive: Vec<String>,
}

impl Vocab {
    /// Seed the extras from a level the editor has just opened, so every authored
    /// off-list value in the file is offered by `c` and by the enum popup from the
    /// first keypress — not only after someone has already stepped off it.
    ///
    /// Purely additive: harvesting twice, or harvesting a level that has no
    /// off-list value anywhere, changes nothing.
    pub fn harvest(&mut self, level: &Level) {
        for n in &level.nodes {
            remember(&crate::entity::NODE_GLYPHS, &mut self.node_glyph, &n.glyph);
        }
        let presets = preset_catalog();
        for m in &level.materials {
            if let Some(r) = m.recipe {
                crate::material::remember_recipe(&mut self.recipe, r);
            }
            let glyph = m.glyph.and_then(char::from_u32);
            remember(&crate::material::GLYPHS, &mut self.material_glyph, &glyph);
            remember(&presets, &mut self.preset, &m.preset.clone());
        }
        let families = emitter_families();
        let glyph_sets = crate::entity::glyph_set_canon();
        let dirs = crate::entity::direction_canon();
        // SYNTHESIZED emitters (`[effects] snow`/`rain` swarms) are not selectable
        // in the editor and are never serialized, so their derived glyph sets are
        // not part of the authored vocabulary.
        for e in level.emitters.iter().filter(|e| !e.synthesized) {
            remember(&families, &mut self.family, &e.family);
            remember(&glyph_sets, &mut self.glyph_set, &e.glyph_set.clone());
            let d = [e.direction.x, e.direction.y, e.direction.z];
            remember_by(&dirs, &mut self.direction, &d, dir_eq);
        }
        let labels = crate::state::event_label_canon();
        for t in &level.triggers {
            if let re_core::TriggerAction::Event { label } = &t.action {
                remember(&labels, &mut self.event_label, label);
            }
        }
        let facings = crate::state::facing_canon();
        remember(&facings, &mut self.facing, &level.player.facing_yaw_col);
        for n in &level.notes {
            remember(&facings, &mut self.facing, &n.facing);
        }
        // Dangling material references: a skin/material binding naming something
        // the level does not define. Cycling one used to erase the name.
        let defined: Vec<String> = level.materials.iter().map(|m| m.name.clone()).collect();
        let mut refs: Vec<&str> = Vec::new();
        for p in &level.portals {
            refs.push(p.skin.as_str());
        }
        for z in &level.overlay.zones {
            refs.push(z.material.0.as_str());
        }
        for l in &level.overlay.lines {
            refs.push(l.material.0.as_str());
        }
        for r in refs {
            if !r.is_empty() {
                remember(&defined, &mut self.material_name, &r.to_string());
            }
        }
        // SOUND FIELDS (`crate::field`): facing/bind/drive values this
        // level authors outside the editor's own canonical pickers — see each
        // string function's doc for exactly which corner is open-ended (a
        // non-cardinal facing vector, a free-form `budget:`/`seq:`/`ring:`
        // target). Seeded for EVERY `field_defs` entry — standalone AND
        // inline-attached — because `state.selected_field` can land on either
        // (the cursor-placement surface reaches only standalone, but direct
        // selection, e.g. the corpus sweep, does not filter by host).
        //
        // Shape-kind (`box`/`capsule`) is deliberately NOT harvested here: it
        // would be a single list shared across every field in the level, so a
        // Box shape on one field would leak "box" into another field's
        // sphere/room option list even though `cycle_shape_kind` never steps a
        // DIFFERENT field into it — see `rowtype.rs`'s `FieldShapeKind` arm,
        // which splices only each field's OWN current value instead.
        let facings: Vec<String> = crate::field::FACING_OPTIONS
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let binds = crate::field::bind_options(level);
        let drives = crate::field::drive_options(level);
        for f in &level.resolved_fields {
            remember(
                &facings,
                &mut self.field_facing,
                &crate::field::facing_to_str(f.cone),
            );
            remember(
                &binds,
                &mut self.field_bind,
                &crate::field::bind_to_str(&f.bind, level),
            );
            remember(
                &drives,
                &mut self.field_drive,
                &crate::field::drive_to_str(&f.drive, level),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_off_list_value_is_spliced_into_the_options_and_a_full_lap_returns() {
        let canon = ['a', 'b', 'c'];
        let extras: Vec<char> = Vec::new();
        let (opts, i) = options_at(&canon, &extras, &'Z');
        assert_eq!(opts, vec!['a', 'b', 'c', 'Z']);
        assert_eq!(
            i, 3,
            "the off-list value is where the cycle actually stands"
        );
    }

    #[test]
    fn remember_makes_the_lap_close() {
        let canon = ['a', 'b', 'c'];
        let mut extras: Vec<char> = Vec::new();
        let mut cur = 'Z';
        // The cycle records the value it steps off, exactly as the real steppers do.
        for _ in 0..4 {
            remember(&canon, &mut extras, &cur);
            cur = step(&canon, &extras, &cur, 1);
        }
        assert_eq!(
            cur, 'Z',
            "four presses over a four-entry cycle is the identity"
        );
    }

    #[test]
    fn remember_is_idempotent_and_leaves_on_list_values_alone() {
        let canon = ['a', 'b'];
        let mut extras: Vec<char> = Vec::new();
        remember(&canon, &mut extras, &'a');
        remember(&canon, &mut extras, &'Z');
        remember(&canon, &mut extras, &'Z');
        assert_eq!(extras, vec!['Z']);
    }

    #[test]
    fn stepping_backwards_from_an_off_list_value_lands_on_the_last_entry() {
        let canon = ['a', 'b', 'c'];
        assert_eq!(step(&canon, &[], &'Z', -1), 'c');
        assert_eq!(step(&canon, &[], &'Z', 1), 'a');
    }

    #[test]
    fn an_empty_canonical_list_is_a_defined_no_op() {
        let canon: [char; 0] = [];
        assert_eq!(step(&canon, &[], &'Z', 1), 'Z');
    }

    #[test]
    fn direction_equality_uses_the_cycles_own_tolerance() {
        assert!(dir_eq(&[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0 - 1e-4]));
        assert!(!dir_eq(&[1.0, 0.0, 0.25], &[1.0, 0.0, 0.0]));
    }

    /// The renderer catalog is the SOURCE, not a copy — and `concrete`, the name
    /// the editor used to offer, is not in it.
    #[test]
    fn the_preset_vocabulary_is_the_renderers_own_catalog() {
        let cat = preset_catalog();
        assert_eq!(cat[0], None, "index 0 is still 'no preset'");
        assert_eq!(
            cat.len(),
            re_render::ascii_shader::ids::CATALOG_NAMES.len() + 1
        );
        for name in re_render::ascii_shader::ids::CATALOG_NAMES {
            assert!(
                cat.contains(&Some((*name).to_string())),
                "catalog missing {name}"
            );
        }
        assert!(
            !cat.contains(&Some("concrete".to_string())),
            "`concrete` names no renderer material and must not be offered"
        );
    }

    #[test]
    fn the_family_vocabulary_is_the_renderers_own_library() {
        let fams = emitter_families();
        assert_eq!(fams.len(), re_render::glyph_family::LIBRARY.len());
        for f in ["haze_o", "state_node", "state_door", "state_trace"] {
            assert!(fams.contains(&f.to_string()), "library missing {f}");
        }
    }
}
