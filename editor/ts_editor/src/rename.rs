//! F2 OBJECT RENAME with reference cascade.
//!
//! `F2` opens a small text-entry prompt (the same modal pattern as Save-As) for
//! the currently-selected named object — from the focused outliner row when the
//! tree is open, else from the active object/material/vox selection. `Enter`
//! commits through ONE choke point, [`rename_object`], which sets the name AND
//! rewrites every reference that is keyed by the name STRING:
//!
//! * **Node** — `[pipe A->B]` endpoints (`Pipe::from`/`to` are strings the
//!   writer emits verbatim), the editor's `spline_pipes` cache (keyed by that
//!   same pair), `[zone_input] node =`, `[trigger] action = activate NAME`,
//!   `[inputs]` rows (`Vec<(String, i32)>`), and `[sequence]`
//!   `observe_names`/`step_names`. Door `target_parts`/`target_off` and
//!   `[emit]` clauses are INDEX-based — the writer derives their names from the
//!   live node list at serialize time, so they follow automatically.
//! * **Material** — `[zone]`/`[line]` `material =` refs, `[portal] skin =`,
//!   and the `[surface]` per-surface bindings (`SurfaceMaterials`). The
//!   1-based ids in `[color_cell_ids]`/painted `side_data` are index-based and
//!   follow automatically.
//! * **Inset** — `[zone] insets = …` membership lists.
//! * Every kind — the COMMENT ANCHOR keys captured here
//!   (`state.preserved`): a `CommentAnchor`/`TrailingComment` is keyed by the
//!   section-header string (`"[node in_a]"`), so a rename rewrites those
//!   headers (incl. the derived `[emit NAME=1]` and `[pipe A->B]` header forms)
//!   or the comments would detach to the file tail on the next save
//!   (`NOTE(rename)` in `state.rs`).
//!
//! Duplicate names within a kind are REFUSED with a status hint (a colliding
//! `[node X]` header would make the reparse guard reject the save anyway — the
//! reparse guard remains the universal backstop under this whole module).
//!
//! Emitters, triggers, notes, and pipes carry NO name in the level model (the
//! writer emits `[trigger]`/`[emitter]` headers without one; a pipe is named by
//! its endpoints), and Door/Spawn/Pedestal are unnamed singletons — F2 on those
//! is a defined no-op with a hint.
//!
//! Pure functions over [`EditorState`] — no I/O, no terminal.
use crate::state::{EditorState, EntityKind, PreservedText, TrailKey};
use crate::tree::{TreeRow, TreeTarget};

/// Which named object the rename prompt targets. Only kinds whose level model
/// actually STORES a name are representable (see the module doc for the rest).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenameTarget {
    /// `level.nodes[i].name`.
    Node(usize),
    /// `level.parts[i].name`.
    Part(usize),
    /// `level.materials[i].name`.
    Material(usize),
    /// `level.voxes[i].name`.
    Vox(usize),
    /// `level.zone_inputs[i].name`.
    ZoneInput(usize),
    /// `level.decals[i].name`.
    Decal(usize),
    /// `level.holos[i].name`.
    Holo(usize),
    /// `level.portals[i].name`.
    Portal(usize),
    /// `level.overlay.zones[i].name`.
    Zone(usize),
    /// `level.overlay.lines[i].name`.
    Line(usize),
    /// `level.overlay.insets[i].name`.
    Inset(usize),
}

/// The open rename prompt: the target object + the editable name buffer.
#[derive(Clone, Debug)]
pub struct RenameState {
    pub target: RenameTarget,
    pub buf: String,
}

/// Open the rename prompt for the current selection (`F2`). Resolution order:
/// the focused outliner row when the tree is open, else the active
/// object/material/vox selection. A kind with no stored name (emitter, trigger,
/// note, pipe, singletons) or no selection is a hinted no-op.
pub fn open(state: &mut EditorState) {
    let target = if state.tree.open {
        target_from_tree(state)
    } else {
        target_from_selection(state)
    };
    let Some(target) = target else {
        state.status = "rename: select a named object first (node/part/material/…)".into();
        return;
    };
    let Some(old) = object_name(state, target) else {
        state.status = "rename: selection out of range".into();
        return;
    };
    state.rename = Some(RenameState { target, buf: old });
    state.status = "rename: type a name  Enter apply  Esc cancel".into();
}

/// Close the prompt without renaming (Esc — called from `quit_on_esc`).
pub fn cancel(state: &mut EditorState) {
    state.rename = None;
    state.status = "rename cancelled".into();
}

/// Modal key handling while the rename prompt is open: printable name-safe
/// chars type, `Backspace` pops, `Enter` commits via [`rename_object`]. Names
/// must stay single parser-safe tokens (they appear inside `[section NAME]`
/// headers and space-separated ref lists), so the accepted set mirrors
/// Save-As: ASCII alphanumerics, `-`, `_`.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_dispatched_key_is_in_the_keymap` reads this file with
/// `include_str!`, scans ONLY that fence for `EditorKeyCode` literals, and
/// asserts each one has a `Context::Rename` binding in `KEYMAP` — otherwise the key
/// would be invisible in the hint bar, the `?` sheet and the `:` palette.
/// Keep the sentinels tight around this body; deleting one does not silently
/// disable the check, it FAILS the test.
pub fn apply_key(state: &mut EditorState, key: re_present::EditorKey) {
    // RENAME-KEYS-BEGIN
    use re_present::EditorKeyCode as E;
    match key.code {
        E::Char(c) => {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                if let Some(r) = state.rename.as_mut() {
                    r.buf.push(c);
                }
            }
        }
        E::Backspace => {
            if let Some(r) = state.rename.as_mut() {
                r.buf.pop();
            }
        }
        E::Enter => {
            let Some(r) = state.rename.take() else { return };
            let new = r.buf.trim().to_string();
            if new.is_empty() {
                // Re-open with the empty buffer kept — the user meant to type.
                state.rename = Some(r);
                state.status = "rename: enter a name".into();
                return;
            }
            match object_name(state, r.target) {
                Some(old) if old == new => {
                    state.status = "rename: unchanged".into();
                }
                Some(_) => match rename_object(state, r.target, &new) {
                    Ok(old) => {
                        state.status = format!("renamed '{old}' → '{new}'");
                    }
                    Err(e) => {
                        // Refused (duplicate) — re-open so the user can fix it.
                        state.rename = Some(r);
                        state.status = e;
                    }
                },
                None => state.status = "rename: selection vanished".into(),
            }
        }
        _ => {}
    }
    // RENAME-KEYS-END
}

/// The outliner row under the tree cursor as a rename target, if it is a named
/// kind.
fn target_from_tree(state: &EditorState) -> Option<RenameTarget> {
    let Some(TreeRow::Object { target, .. }) = crate::tree::current_row(state) else {
        return None;
    };
    match target {
        TreeTarget::Node(i) => Some(RenameTarget::Node(i)),
        TreeTarget::Part(i) => Some(RenameTarget::Part(i)),
        TreeTarget::Vox(i) => Some(RenameTarget::Vox(i)),
        TreeTarget::ZoneInput(i) => Some(RenameTarget::ZoneInput(i)),
        TreeTarget::WallDecal(i) => Some(RenameTarget::Decal(i)),
        TreeTarget::Holo(i) => Some(RenameTarget::Holo(i)),
        TreeTarget::Portal(i) => Some(RenameTarget::Portal(i)),
        TreeTarget::Zone(i) => Some(RenameTarget::Zone(i)),
        TreeTarget::Line(i) => Some(RenameTarget::Line(i)),
        TreeTarget::Inset(i) => Some(RenameTarget::Inset(i)),
        // No stored name: emitters/triggers/notes/pipes + the singletons.
        TreeTarget::Emitter(_)
        | TreeTarget::Trigger(_)
        | TreeTarget::Note(_)
        | TreeTarget::Pipe(_)
        | TreeTarget::Singleton(_) => None,
    }
}

/// The active object/material/vox selection as a rename target, if named.
fn target_from_selection(state: &EditorState) -> Option<RenameTarget> {
    if state.material_mode {
        return crate::material::sel_index(state).map(RenameTarget::Material);
    }
    if state.vox_mode {
        return state.selected_vox.map(RenameTarget::Vox);
    }
    match state.place? {
        EntityKind::Node => state.selected_node.map(RenameTarget::Node),
        EntityKind::Part => state.selected_part.map(RenameTarget::Part),
        EntityKind::ZoneInput => state.selected_zone_input.map(RenameTarget::ZoneInput),
        EntityKind::WallDecal => state.selected_decal.map(RenameTarget::Decal),
        EntityKind::Holo => state.selected_holo.map(RenameTarget::Holo),
        EntityKind::Portal => state.selected_portal.map(RenameTarget::Portal),
        EntityKind::Zone => state.selected_zone.map(RenameTarget::Zone),
        EntityKind::Line => state.selected_line.map(RenameTarget::Line),
        EntityKind::Inset => state.selected_inset.map(RenameTarget::Inset),
        // Unnamed kinds (see module doc). `Field` DOES carry a stored name
        // (`FieldDef::name`), but F2's reference-cascade rename (rewriting
        // every OTHER field's `bind =`/`drive =` that targets it by name) is
        // out of this pass's scope — see `crate::field`'s module doc. F2 on
        // a Field is a defined no-op with a hint, same as the other kinds
        // below; renaming one today means hand-editing the `.lvl` text.
        EntityKind::Door
        | EntityKind::Spawn
        | EntityKind::Pedestal
        | EntityKind::Emitter
        | EntityKind::Trigger
        | EntityKind::Pipe
        | EntityKind::Note
        | EntityKind::Field => None,
    }
}

/// The target's CURRENT name (bound-checked; `None` when the index went stale).
pub fn object_name(state: &EditorState, target: RenameTarget) -> Option<String> {
    let l = &state.level;
    match target {
        RenameTarget::Node(i) => l.nodes.get(i).map(|n| n.name.clone()),
        RenameTarget::Part(i) => l.parts.get(i).map(|p| p.name.clone()),
        RenameTarget::Material(i) => l.materials.get(i).map(|m| m.name.clone()),
        RenameTarget::Vox(i) => l.voxes.get(i).map(|v| v.name.clone()),
        RenameTarget::ZoneInput(i) => l.zone_inputs.get(i).map(|z| z.name.clone()),
        RenameTarget::Decal(i) => l.decals.get(i).map(|d| d.name.clone()),
        RenameTarget::Holo(i) => l.holos.get(i).map(|h| h.name.clone()),
        RenameTarget::Portal(i) => l.portals.get(i).map(|p| p.name.clone()),
        RenameTarget::Zone(i) => l.overlay.zones.get(i).map(|z| z.name.clone()),
        RenameTarget::Line(i) => l.overlay.lines.get(i).map(|ln| ln.name.clone()),
        RenameTarget::Inset(i) => l.overlay.insets.get(i).map(|n| n.name.clone()),
    }
}

/// Is `name` already taken by a DIFFERENT object of the same kind? (The
/// duplicate-refusal predicate — colliding section headers would corrupt
/// name-keyed references and the reparse guard would refuse the save.)
fn name_taken(state: &EditorState, target: RenameTarget, name: &str) -> bool {
    let l = &state.level;
    match target {
        RenameTarget::Node(i) => l
            .nodes
            .iter()
            .enumerate()
            .any(|(j, n)| j != i && n.name == name),
        RenameTarget::Part(i) => l
            .parts
            .iter()
            .enumerate()
            .any(|(j, p)| j != i && p.name == name),
        RenameTarget::Material(i) => l
            .materials
            .iter()
            .enumerate()
            .any(|(j, m)| j != i && m.name == name),
        RenameTarget::Vox(i) => l
            .voxes
            .iter()
            .enumerate()
            .any(|(j, v)| j != i && v.name == name),
        RenameTarget::ZoneInput(i) => l
            .zone_inputs
            .iter()
            .enumerate()
            .any(|(j, z)| j != i && z.name == name),
        RenameTarget::Decal(i) => l
            .decals
            .iter()
            .enumerate()
            .any(|(j, d)| j != i && d.name == name),
        RenameTarget::Holo(i) => l
            .holos
            .iter()
            .enumerate()
            .any(|(j, h)| j != i && h.name == name),
        RenameTarget::Portal(i) => l
            .portals
            .iter()
            .enumerate()
            .any(|(j, p)| j != i && p.name == name),
        RenameTarget::Zone(i) => l
            .overlay
            .zones
            .iter()
            .enumerate()
            .any(|(j, z)| j != i && z.name == name),
        RenameTarget::Line(i) => l
            .overlay
            .lines
            .iter()
            .enumerate()
            .any(|(j, ln)| j != i && ln.name == name),
        RenameTarget::Inset(i) => l
            .overlay
            .insets
            .iter()
            .enumerate()
            .any(|(j, n)| j != i && n.name == name),
    }
}

/// THE rename choke point: validate, snapshot undo, set the name, and rewrite
/// every string-keyed reference (see the module doc for the exhaustive per-kind
/// cascade). Returns the OLD name on success, a status-ready error on refusal.
///
/// Undo covers `(map, level)` — the level-side cascade is fully undoable. The
/// editor-side caches outside the snapshot (`spline_pipes`, `preserved` comment
/// anchors) keep the NEW name after an undo; a stale anchor is never lost (the
/// io.rs re-emitter demotes unmatched anchors to file-tail comments) and the
/// The reparse guard backstops the save either way.
pub fn rename_object(
    state: &mut EditorState,
    target: RenameTarget,
    new: &str,
) -> Result<String, String> {
    let Some(old) = object_name(state, target) else {
        return Err("rename: selection out of range".into());
    };
    if name_taken(state, target, new) {
        return Err(format!(
            "rename refused: '{new}' already exists in this kind"
        ));
    }
    state.push_undo();
    let l = &mut state.level;
    match target {
        RenameTarget::Node(i) => {
            l.nodes[i].name = new.to_string();
            // [pipe A->B] endpoints are STRINGS on the pipe (the writer emits
            // them verbatim in the header — a stale name is a hard parse error
            // on reload, which the reparse guard would refuse).
            for p in &mut l.pipes {
                if p.from == old {
                    p.from = new.to_string();
                }
                if p.to == old {
                    p.to = new.to_string();
                }
            }
            // [zone_input] node = NAME.
            for z in &mut l.zone_inputs {
                if z.node == old {
                    z.node = new.to_string();
                }
            }
            // [trigger] action = activate NAME.
            for t in &mut l.triggers {
                if let re_level::TriggerAction::ActivateNode { name, .. } = &mut t.action {
                    if *name == old {
                        *name = new.to_string();
                    }
                }
            }
            // [inputs] NAME = val rows.
            for (n, _) in &mut l.inputs {
                if *n == old {
                    *n = new.to_string();
                }
            }
            // [sequence] observe = NAME... / step = NAME... (H1's cascade core).
            for n in &mut l.sequence.observe_names {
                if *n == old {
                    *n = new.to_string();
                }
            }
            for step in &mut l.sequence.step_names {
                for n in step {
                    if *n == old {
                        *n = new.to_string();
                    }
                }
            }
            // Editor-side cache: the spline-mode set is keyed by (from, to).
            let renamed: std::collections::HashSet<(String, String)> = state
                .spline_pipes
                .drain()
                .map(|(f, t)| {
                    (
                        if f == old { new.to_string() } else { f },
                        if t == old { new.to_string() } else { t },
                    )
                })
                .collect();
            state.spline_pipes = renamed;
            // Comment anchors: [node X], the derived [emit X=1], and the
            // [pipe A->B] headers that embed the node name.
            rewrite_headers(&mut state.preserved, |h| node_header_renamed(h, &old, new));
            rewrite_inputs_keys(&mut state.preserved, &old, new);
        }
        RenameTarget::Part(i) => {
            // [emit] parts lists and [hierarchy] links are index-based — the
            // writer derives their names from the live part list, so only the
            // part's own name (and its comment anchors) change.
            l.parts[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "part", &old, new)
            });
        }
        RenameTarget::Material(i) => {
            l.materials[i].name = new.to_string();
            // [zone]/[line] material = NAME (Scalable Material Overlay refs).
            for z in &mut l.overlay.zones {
                if z.material.0 == old {
                    z.material.0 = new.to_string();
                }
            }
            for ln in &mut l.overlay.lines {
                if ln.material.0 == old {
                    ln.material.0 = new.to_string();
                }
            }
            // [portal] skin = NAME (an optional material tint).
            for p in &mut l.portals {
                if p.skin == old {
                    p.skin = new.to_string();
                }
            }
            // [surface] per-surface bindings.
            let sm = &mut l.surface_materials;
            for slot in [
                &mut sm.wall_solid,
                &mut sm.wall_trace,
                &mut sm.wall_matrix,
                &mut sm.floor,
                &mut sm.ceiling,
            ] {
                if slot.as_deref() == Some(old.as_str()) {
                    *slot = Some(new.to_string());
                }
            }
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "material", &old, new)
            });
        }
        RenameTarget::Vox(i) => {
            l.voxes[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "vox", &old, new)
            });
        }
        RenameTarget::ZoneInput(i) => {
            l.zone_inputs[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "zone_input", &old, new)
            });
        }
        RenameTarget::Decal(i) => {
            l.decals[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "wall_decal", &old, new)
            });
        }
        RenameTarget::Holo(i) => {
            l.holos[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "holo", &old, new)
            });
        }
        RenameTarget::Portal(i) => {
            l.portals[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "portal", &old, new)
            });
        }
        RenameTarget::Zone(i) => {
            l.overlay.zones[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "zone", &old, new)
            });
        }
        RenameTarget::Line(i) => {
            l.overlay.lines[i].name = new.to_string();
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "line", &old, new)
            });
        }
        RenameTarget::Inset(i) => {
            l.overlay.insets[i].name = new.to_string();
            // [zone] insets = NAME... membership lists reference insets by name.
            for z in &mut l.overlay.zones {
                for r in &mut z.insets {
                    if r.0 == old {
                        r.0 = new.to_string();
                    }
                }
            }
            rewrite_headers(&mut state.preserved, |h| {
                section_header_renamed(h, "inset", &old, new)
            });
        }
    }
    state.mark_dirty();
    Ok(old)
}

/// Rewrite every comment-anchor / trailing-comment section header the mapper
/// recognizes (anchors are keyed by the header STRING — see
/// `NOTE(rename)` in `state.rs`).
fn rewrite_headers(p: &mut PreservedText, renamed: impl Fn(&str) -> Option<String>) {
    for a in &mut p.anchors {
        if let Some(nh) = renamed(&a.header) {
            a.header = nh;
        }
    }
    for t in &mut p.trailing {
        if let Some(nh) = renamed(&t.header) {
            t.header = nh;
        }
    }
}

/// `"[<kw> OLD]"` → `"[<kw> NEW]"` for a simple named-section header.
fn section_header_renamed(header: &str, kw: &str, old: &str, new: &str) -> Option<String> {
    if header == format!("[{kw} {old}]") {
        Some(format!("[{kw} {new}]"))
    } else {
        None
    }
}

/// The NODE header forms a rename must chase: its own `[node X]`, the derived
/// `[emit X=1]`, and `[pipe A->B]` (either endpoint, ASCII `->` or unicode `→`).
fn node_header_renamed(header: &str, old: &str, new: &str) -> Option<String> {
    if let Some(nh) = section_header_renamed(header, "node", old, new) {
        return Some(nh);
    }
    if header == format!("[emit {old}=1]") {
        return Some(format!("[emit {new}=1]"));
    }
    let inner = header
        .strip_prefix("[pipe ")
        .and_then(|s| s.strip_suffix(']'))?;
    let (from, to, arrow) = inner
        .split_once("->")
        .map(|(f, t)| (f.trim(), t.trim(), "->"))
        .or_else(|| {
            inner
                .split_once('→')
                .map(|(f, t)| (f.trim(), t.trim(), "→"))
        })?;
    if from != old && to != old {
        return None;
    }
    let nf = if from == old { new } else { from };
    let nt = if to == old { new } else { to };
    Some(format!("[pipe {nf}{arrow}{nt}]"))
}

/// `[inputs]` interior comment anchors/trailing comments are keyed by the NODE
/// name itself (the row is `NAME = val`), so a node rename rewrites those keys.
fn rewrite_inputs_keys(p: &mut PreservedText, old: &str, new: &str) {
    for a in &mut p.anchors {
        if a.header == "[inputs]" && a.key.as_deref() == Some(old) {
            a.key = Some(new.to_string());
        }
    }
    for t in &mut p.trailing {
        if t.header == "[inputs]" {
            if let TrailKey::Key(k) = &mut t.key {
                if k == old {
                    *k = new.to_string();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{CommentAnchor, TrailingComment};
    use re_present::{EditorKey, EditorKeyCode as E};

    fn key(code: E) -> EditorKey {
        EditorKey { code, shift: false }
    }

    /// A level with two nodes and every node-name-keyed reference kind.
    fn node_ref_state() -> EditorState {
        let mut s = EditorState::new_empty(12, 12, 4);
        let a = s.level.push_node(1, 1, 0);
        let b = s.level.push_node(5, 5, 0);
        // `[zone_input]`/`[trigger] activate` targets MUST be input nodes
        // (re_level finalize enforces it), so make `a` one.
        s.level.nodes[a].is_input = true;
        let a_name = s.level.nodes[a].name.clone(); // "node0"
        let b_name = s.level.nodes[b].name.clone();
        // Pipe a -> b (string endpoints).
        s.level.pipes.push(re_level::Pipe {
            from: a_name.clone(),
            to: b_name.clone(),
            from_node: a,
            to_node: b,
            path: vec![(1, 1, 0), (2, 1, 0)],
            ..Default::default()
        });
        s.spline_pipes.insert((a_name.clone(), b_name.clone()));
        // Zone input driving a.
        s.level.zone_inputs.push(re_level::ZoneInput {
            name: "zi0".into(),
            node: a_name.clone(),
            node_index: a,
            x0: 0,
            y0: 0,
            z0: 0,
            x1: 1,
            y1: 1,
            z1: 1,
        });
        // Trigger activating a.
        s.level.triggers.push(re_level::Trigger {
            action: re_level::TriggerAction::activate_node(a_name.clone()),
            ..Default::default()
        });
        // [inputs] row.
        s.level.inputs.push((a_name.clone(), 1));
        // [sequence] observe/step names.
        s.level.sequence.observe_names = vec![a_name.clone(), b_name];
        s.level.sequence.step_names = vec![vec![a_name], vec![]];
        s
    }

    #[test]
    fn rename_node_cascades_sequence_names() {
        let mut s = node_ref_state();
        let old = s.level.nodes[0].name.clone();
        rename_object(&mut s, RenameTarget::Node(0), "furnace").expect("renames");
        assert_eq!(s.level.nodes[0].name, "furnace");
        assert_eq!(s.level.sequence.observe_names[0], "furnace");
        assert_eq!(s.level.sequence.step_names[0][0], "furnace");
        assert!(
            !s.level.sequence.observe_names.contains(&old),
            "no stale name in observe"
        );
        assert!(s.dirty, "rename marks dirty");
    }

    #[test]
    fn rename_node_cascades_pipe_endpoints_and_spline_cache() {
        let mut s = node_ref_state();
        let b_name = s.level.nodes[1].name.clone();
        rename_object(&mut s, RenameTarget::Node(0), "furnace").expect("renames");
        assert_eq!(s.level.pipes[0].from, "furnace", "pipe endpoint follows");
        assert_eq!(s.level.pipes[0].to, b_name);
        assert!(
            s.spline_pipes.contains(&("furnace".to_string(), b_name)),
            "editor spline cache re-keyed"
        );
        assert_eq!(s.spline_pipes.len(), 1);
    }

    #[test]
    fn rename_node_cascades_zone_input_trigger_and_inputs() {
        let mut s = node_ref_state();
        rename_object(&mut s, RenameTarget::Node(0), "furnace").expect("renames");
        assert_eq!(
            s.level.zone_inputs[0].node, "furnace",
            "[zone_input] node follows"
        );
        match &s.level.triggers[0].action {
            re_level::TriggerAction::ActivateNode { name, .. } => {
                assert_eq!(name, "furnace", "[trigger] activate follows");
            }
            other => panic!("unexpected action {other:?}"),
        }
        assert_eq!(s.level.inputs[0].0, "furnace", "[inputs] row follows");
    }

    #[test]
    fn rename_material_cascades_overlay_portal_and_surface_refs() {
        let mut s = EditorState::new_empty(8, 8, 4);
        let mi = s.level.push_material();
        let old = s.level.materials[mi].name.clone();
        // [zone] + [line] material refs.
        s.level.overlay.zones.push(re_level::OverlayZone {
            name: "z0".into(),
            material: re_core::MaterialId(old.clone()),
            ..Default::default()
        });
        s.level.overlay.lines.push(re_level::OverlayLine {
            name: "l0".into(),
            material: re_core::LineMaterialId(old.clone()),
            ..Default::default()
        });
        // [portal] skin.
        s.level.portals.push(re_level::LevelPortal {
            name: "p0".into(),
            skin: old.clone(),
            ..Default::default()
        });
        // [surface] binding.
        s.level.surface_materials.wall_solid = Some(old);

        rename_object(&mut s, RenameTarget::Material(mi), "rusty").expect("renames");
        assert_eq!(s.level.materials[mi].name, "rusty");
        assert_eq!(
            s.level.overlay.zones[0].material.0, "rusty",
            "[zone] material follows"
        );
        assert_eq!(
            s.level.overlay.lines[0].material.0, "rusty",
            "[line] material follows"
        );
        assert_eq!(s.level.portals[0].skin, "rusty", "[portal] skin follows");
        assert_eq!(
            s.level.surface_materials.wall_solid.as_deref(),
            Some("rusty"),
            "[surface] binding follows"
        );
    }

    #[test]
    fn rename_inset_cascades_zone_membership() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.level.overlay.insets.push(re_level::OverlayInset {
            name: "carving".into(),
            ..Default::default()
        });
        s.level.overlay.zones.push(re_level::OverlayZone {
            name: "z0".into(),
            insets: vec![re_core::InsetId("carving".into())],
            ..Default::default()
        });

        rename_object(&mut s, RenameTarget::Inset(0), "relief").expect("renames");
        assert_eq!(s.level.overlay.insets[0].name, "relief");
        assert_eq!(
            s.level.overlay.zones[0].insets[0].0, "relief",
            "[zone] insets membership follows"
        );
    }

    #[test]
    fn rename_refuses_duplicate_within_kind() {
        let mut s = node_ref_state();
        let b_name = s.level.nodes[1].name.clone();
        let a_name = s.level.nodes[0].name.clone();
        let err = rename_object(&mut s, RenameTarget::Node(0), &b_name)
            .expect_err("duplicate must refuse");
        assert!(err.contains("refused"), "refusal message: {err}");
        assert_eq!(s.level.nodes[0].name, a_name, "name unchanged on refusal");
        // A DIFFERENT kind may reuse the name (per-kind namespaces).
        s.level.parts.push(crate::part::new_part(
            "widget".into(),
            1,
            1,
            1,
            (0, 0, 0),
            0xFF_FFFF,
        ));
        rename_object(&mut s, RenameTarget::Part(0), &a_name).expect("cross-kind reuse is allowed");
    }

    /// Comment anchors keyed by section-header strings follow the
    /// rename — the plain `[node X]` header, the derived `[emit X=1]` and
    /// `[pipe A->B]` forms, and the `[inputs]` interior key.
    #[test]
    fn rename_node_rewrites_comment_anchor_keys() {
        let mut s = node_ref_state();
        let a = s.level.nodes[0].name.clone();
        let b = s.level.nodes[1].name.clone();
        s.preserved.anchors.push(CommentAnchor {
            header: format!("[node {a}]"),
            key: None,
            lines: vec!["# the furnace node".into()],
        });
        s.preserved.anchors.push(CommentAnchor {
            header: format!("[emit {a}=1]"),
            key: None,
            lines: vec!["# lights the furnace part".into()],
        });
        s.preserved.anchors.push(CommentAnchor {
            header: format!("[pipe {a}->{b}]"),
            key: None,
            lines: vec!["# main power line".into()],
        });
        s.preserved.anchors.push(CommentAnchor {
            header: "[inputs]".into(),
            key: Some(a.clone()),
            lines: vec!["# start hot".into()],
        });
        s.preserved.trailing.push(TrailingComment {
            header: format!("[node {a}]"),
            key: TrailKey::Header,
            suffix: " # hub".into(),
        });
        s.preserved.trailing.push(TrailingComment {
            header: "[inputs]".into(),
            key: TrailKey::Key(a),
            suffix: " # initial".into(),
        });

        rename_object(&mut s, RenameTarget::Node(0), "furnace").expect("renames");
        assert_eq!(s.preserved.anchors[0].header, "[node furnace]");
        assert_eq!(s.preserved.anchors[1].header, "[emit furnace=1]");
        assert_eq!(
            s.preserved.anchors[2].header,
            format!("[pipe furnace->{b}]")
        );
        assert_eq!(s.preserved.anchors[3].key.as_deref(), Some("furnace"));
        assert_eq!(s.preserved.trailing[0].header, "[node furnace]");
        assert_eq!(s.preserved.trailing[1].key, TrailKey::Key("furnace".into()));
    }

    /// END-TO-END: rename a node that every string-keyed reference kind points
    /// at, serialize through the REAL save path, and reparse — the file loads
    /// clean, carries the new name everywhere, and has zero stale occurrences.
    #[test]
    fn rename_then_serialize_reparses_clean_with_no_stale_name() {
        let mut s = node_ref_state();
        let old = s.level.nodes[0].name.clone();
        rename_object(&mut s, RenameTarget::Node(0), "furnace").expect("renames");
        let text = crate::io::capture_and_serialize(&mut s);
        assert!(
            !text.contains(&old),
            "no stale '{old}' anywhere in the serialized file:\n{text}"
        );
        assert!(text.contains("furnace"), "new name present");
        let reloaded =
            re_level::load_string(&text, "rename-rt.lvl").expect("reparses clean (H1 backstop)");
        assert!(reloaded.nodes.iter().any(|n| n.name == "furnace"));
        assert_eq!(reloaded.pipes[0].from, "furnace");
        assert_eq!(reloaded.zone_inputs[0].node, "furnace");
        assert_eq!(reloaded.sequence.observe_names[0], "furnace");
    }

    /// The F2 modal flow: open seeds the buffer with the current name; typed
    /// chars edit it; Enter commits the rename through the cascade.
    #[test]
    fn f2_prompt_flow_renames_selected_node() {
        let mut s = node_ref_state();
        s.place = Some(EntityKind::Node);
        s.selected_node = Some(0);
        let old = s.level.nodes[0].name.clone();
        open(&mut s);
        let r = s.rename.as_ref().expect("prompt open");
        assert_eq!(r.buf, old, "buffer seeded with the current name");
        // Clear the seed, type a new name, commit.
        for _ in 0..old.len() {
            apply_key(&mut s, key(E::Backspace));
        }
        for c in "core".chars() {
            apply_key(&mut s, key(E::Char(c)));
        }
        apply_key(&mut s, key(E::Enter));
        assert!(s.rename.is_none(), "prompt closed on commit");
        assert_eq!(s.level.nodes[0].name, "core");
        assert_eq!(s.level.sequence.observe_names[0], "core", "cascade ran");
        assert!(s.status.contains("renamed"), "status: {}", s.status);
    }

    /// Unsafe characters are filtered while typing (names live inside section
    /// headers and space-separated ref lists), and a duplicate commit re-opens
    /// the prompt with the refusal message.
    #[test]
    fn f2_prompt_filters_chars_and_reopens_on_duplicate() {
        let mut s = node_ref_state();
        let b_name = s.level.nodes[1].name.clone();
        s.place = Some(EntityKind::Node);
        s.selected_node = Some(0);
        open(&mut s);
        let seed_len = s.rename.as_ref().unwrap().buf.len();
        for _ in 0..seed_len {
            apply_key(&mut s, key(E::Backspace));
        }
        for c in b_name.chars() {
            apply_key(&mut s, key(E::Char(c)));
        }
        apply_key(&mut s, key(E::Char(' '))); // filtered
        apply_key(&mut s, key(E::Char(']'))); // filtered
        assert_eq!(
            s.rename.as_ref().unwrap().buf,
            b_name,
            "unsafe chars filtered"
        );
        apply_key(&mut s, key(E::Enter));
        assert!(s.rename.is_some(), "duplicate refusal re-opens the prompt");
        assert!(s.status.contains("refused"), "status: {}", s.status);
    }

    /// F2 on an unnamed kind (emitter/trigger/note/pipe/singleton) is a hinted
    /// no-op — those carry no name in the level model.
    #[test]
    fn f2_on_unnamed_kind_is_a_hinted_noop() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.place = Some(EntityKind::Emitter);
        s.selected_emitter = Some(0);
        open(&mut s);
        assert!(s.rename.is_none(), "no prompt for an unnamed kind");
        assert!(s.status.contains("rename"), "status hints: {}", s.status);
    }
}
