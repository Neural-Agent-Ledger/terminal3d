//! OBJECT NESTING / PARENTING for the editor.
//!
//! Pure functions over [`EditorState`] — no I/O, no terminal. The nesting graph
//! itself lives in `level.hierarchy` (a `Vec<(child HierRef, parent HierRef)>`,
//! owned by `re_level`), so it round-trips through save/reload AND through the
//! editor's snapshot undo (which clones `level`). This module is the bridge
//! between the editor's [`TreeTarget`] selection model and that persisted graph,
//! plus the GROUP-MOVE that makes "moving a parent moves its descendants" true.
//!
//! ## Group-move (the core behaviour)
//!
//! Object position edits are scattered across many paths (the focused-field
//! `+`/`-` editor for door/node/pedestal/spawn/emitter/trigger, the tree
//! jump-move, …). Rather than intercept each one, the group-move is implemented
//! as a SNAPSHOT → EDIT → DIFF wrapper:
//!
//! 1. [`snapshot_positions`] records every object's absolute position BEFORE the
//!    edit.
//! 2. the edit runs (any path that mutates a position).
//! 3. [`apply_group_moves`] compares the new positions to the snapshot; for every
//!    object that MOVED, it adds the SAME delta to all of that object's transitive
//!    descendants. Multi-level nesting falls out for free: a grandparent move
//!    shifts its child (directly) and the child's own descendants (because the
//!    descendant set is transitive).
//!
//! A descendant that is ALSO a moved root in the same edit is skipped from the
//! propagation (its own explicit move wins), so a single edit never double-counts.
//! In practice only one object moves per keystroke, so this is the simple case.
use re_level::{HierKind, HierRef};

use crate::state::{EditorState, EntityKind};
use crate::tree::TreeTarget;

/// Map an editor [`TreeTarget`] to the persisted [`HierRef`]. Every selectable
/// tree object has a uniform reference.
pub fn target_to_ref(target: TreeTarget) -> HierRef {
    match target {
        TreeTarget::Node(i) => HierRef::new(HierKind::Node, i),
        TreeTarget::Emitter(i) => HierRef::new(HierKind::Emitter, i),
        TreeTarget::Trigger(i) => HierRef::new(HierKind::Trigger, i),
        TreeTarget::Vox(i) => HierRef::new(HierKind::Vox, i),
        TreeTarget::Singleton(kind) => HierRef::new(singleton_hier_kind(kind), 0),
        // Pipes / ZoneInputs / Notes / WallDecals / Parts are NOT part of the
        // object-nesting model (no matching `HierKind`); a sentinel out-of-range Node
        // ref keeps this total without ever matching a real hierarchy entry. Reparent
        // is refused for these rows separately (they report no nestable target).
        TreeTarget::Pipe(_)
        | TreeTarget::ZoneInput(_)
        | TreeTarget::Note(_)
        | TreeTarget::WallDecal(_)
        | TreeTarget::Holo(_)
        | TreeTarget::Portal(_)
        | TreeTarget::Zone(_)
        | TreeTarget::Line(_)
        | TreeTarget::Inset(_)
        | TreeTarget::Part(_) => HierRef::new(HierKind::Node, usize::MAX),
    }
}

/// Map a persisted [`HierRef`] back to an editor [`TreeTarget`]. The inverse of
/// [`target_to_ref`]; used by the tree to re-select a parent/child after a
/// hierarchy edit (and to round-trip the mapping in tests).
#[cfg(test)]
pub fn ref_to_target(r: HierRef) -> TreeTarget {
    match r.kind {
        HierKind::Node => TreeTarget::Node(r.index),
        HierKind::Emitter => TreeTarget::Emitter(r.index),
        HierKind::Trigger => TreeTarget::Trigger(r.index),
        HierKind::Vox => TreeTarget::Vox(r.index),
        HierKind::Door => TreeTarget::Singleton(EntityKind::Door),
        HierKind::Spawn => TreeTarget::Singleton(EntityKind::Spawn),
        HierKind::Pedestal => TreeTarget::Singleton(EntityKind::Pedestal),
    }
}

/// Map an [`EntityKind`] singleton to its [`HierKind`]. Only Door/Spawn/Pedestal
/// are singletons; Node/Emitter/Trigger are multi-instance and never reach here.
fn singleton_hier_kind(kind: EntityKind) -> HierKind {
    match kind {
        EntityKind::Door => HierKind::Door,
        EntityKind::Spawn => HierKind::Spawn,
        EntityKind::Pedestal => HierKind::Pedestal,
        EntityKind::Node => HierKind::Node,
        EntityKind::Emitter => HierKind::Emitter,
        EntityKind::Trigger => HierKind::Trigger,
        // Pipes / ZoneInputs / Notes / WallDecals / Parts / Fields are not
        // singletons and not part of the hierarchy; never reached via a
        // singleton path.
        EntityKind::Pipe
        | EntityKind::ZoneInput
        | EntityKind::Note
        | EntityKind::WallDecal
        | EntityKind::Holo
        | EntityKind::Portal
        | EntityKind::Zone
        | EntityKind::Line
        | EntityKind::Inset
        | EntityKind::Part
        | EntityKind::Field => HierKind::Node,
    }
}

/// Short readable label for a [`HierRef`] (status hints / tree breadcrumbs).
pub fn ref_label(state: &EditorState, r: HierRef) -> String {
    match r.kind {
        HierKind::Node => state
            .level
            .nodes
            .get(r.index)
            .map(|n| format!("node '{}'", n.name))
            .unwrap_or_else(|| format!("node #{}", r.index)),
        HierKind::Emitter => format!("emitter #{}", r.index),
        HierKind::Trigger => format!("trigger #{}", r.index),
        HierKind::Vox => format!("vox #{}", r.index),
        HierKind::Door => "door".into(),
        HierKind::Spawn => "spawn".into(),
        HierKind::Pedestal => "pedestal".into(),
    }
}

// ───────────────────────────── positions ─────────────────────────────

/// The absolute position of an object as a float `(x, y, z)`. Integer kinds
/// report their cell as a float; an emitter reports its `World` origin (a `Screen`
/// origin reports `(col, row, 0)` and is treated as un-translatable). A trigger
/// reports its region MIN corner. `None` ⇒ the object is gone (stale index).
pub fn position_of(state: &EditorState, r: HierRef) -> Option<(f32, f32, f32)> {
    let lvl = &state.level;
    Some(match r.kind {
        HierKind::Node => {
            let n = lvl.nodes.get(r.index)?;
            (n.x as f32, n.y as f32, n.z as f32)
        }
        HierKind::Emitter => {
            let e = lvl.emitters.get(r.index)?;
            match e.origin {
                re_world::EmitterOrigin::World(p) => (p.x, p.y, p.z),
                re_world::EmitterOrigin::Screen { col, row } => (col, row, 0.0),
            }
        }
        HierKind::Trigger => {
            let t = lvl.triggers.get(r.index)?;
            match t.region {
                re_core::Region::Box { min, .. } => (min.0 as f32, min.1 as f32, min.2 as f32),
                _ => return None,
            }
        }
        HierKind::Vox => {
            let v = lvl.voxes.get(r.index)?;
            (v.off_x as f32, v.off_y as f32, v.off_z as f32)
        }
        HierKind::Door => (lvl.door.x as f32, lvl.door.y as f32, lvl.door.z as f32),
        HierKind::Spawn => (
            lvl.player.x as f32,
            lvl.player.y as f32,
            lvl.player.z as f32,
        ),
        HierKind::Pedestal => (
            lvl.pedestal.cx as f32,
            lvl.pedestal.cy as f32,
            lvl.pedestal.cz as f32,
        ),
    })
}

/// Add a float `delta` to an object's absolute position. Integer kinds round the
/// delta to whole cells (the editor only ever moves them by whole cells, so this
/// is exact in practice); an emitter `World` origin takes the float delta. A
/// `Screen` emitter origin and a non-Box trigger are left unchanged. No clamping
/// (the parent's own move was already clamped; a descendant rides along and a
/// stray out-of-room cell is harmless — `build_map` clips it).
pub fn translate(state: &mut EditorState, r: HierRef, delta: (f32, f32, f32)) {
    let (dx, dy, dz) = delta;
    let lvl = &mut state.level;
    let (ix, iy, iz) = (dx.round() as i32, dy.round() as i32, dz.round() as i32);
    match r.kind {
        HierKind::Node => {
            if let Some(n) = lvl.nodes.get_mut(r.index) {
                n.x += ix;
                n.y += iy;
                n.z += iz;
            }
        }
        HierKind::Emitter => {
            if let Some(e) = lvl.emitters.get_mut(r.index) {
                if let re_world::EmitterOrigin::World(mut p) = e.origin {
                    p.x += dx;
                    p.y += dy;
                    p.z += dz;
                    e.origin = re_world::EmitterOrigin::World(p);
                    e.validate();
                }
            }
        }
        HierKind::Trigger => {
            if let Some(t) = lvl.triggers.get_mut(r.index) {
                if let re_core::Region::Box { min, max } = t.region {
                    t.region = re_core::Region::Box {
                        min: (min.0 + ix, min.1 + iy, min.2 + iz),
                        max: (max.0 + ix, max.1 + iy, max.2 + iz),
                    };
                }
            }
        }
        HierKind::Vox => {
            if let Some(v) = lvl.voxes.get_mut(r.index) {
                v.off_x += ix;
                v.off_y += iy;
                v.off_z += iz;
            }
        }
        HierKind::Door => {
            lvl.door.x += ix;
            lvl.door.y += iy;
            lvl.door.z += iz;
        }
        HierKind::Spawn => {
            lvl.player.x += ix;
            lvl.player.y += iy;
            lvl.player.z += iz;
        }
        HierKind::Pedestal => {
            lvl.pedestal.cx += ix;
            lvl.pedestal.cy += iy;
            lvl.pedestal.cz += iz;
        }
    }
}

// ───────────────────────────── group-move ─────────────────────────────

/// Every object that can be a hierarchy node, paired with its current position.
/// The snapshot domain: only objects that PARTICIPATE in the hierarchy (are a
/// parent or a child of some link) need watching, but enumerating all of them is
/// cheap and keeps the diff trivially correct. A gone/Screen/non-Box object is
/// omitted (no position to diff).
pub fn snapshot_positions(state: &EditorState) -> Vec<(HierRef, (f32, f32, f32))> {
    let mut out = Vec::new();
    // Only the refs that appear in some link can possibly need group-move; a
    // hierarchy-less level produces an EMPTY snapshot, so `apply_group_moves` is a
    // no-op (and the wrapper is byte-for-byte transparent).
    let mut refs: Vec<HierRef> = Vec::new();
    for (c, p) in &state.level.hierarchy {
        if !refs.contains(c) {
            refs.push(*c);
        }
        if !refs.contains(p) {
            refs.push(*p);
        }
    }
    for r in refs {
        if let Some(pos) = position_of(state, r) {
            out.push((r, pos));
        }
    }
    out
}

/// Compare current positions against `before`; for every object that MOVED, apply
/// the same delta to all of its transitive descendants. Call AFTER an edit that
/// may have moved an object, with the snapshot taken before it.
///
/// Multi-level nesting: descendants are transitive, so a grandparent's move shifts
/// the child AND the grandchild in one pass. A descendant that is ITSELF a moved
/// root in the same edit keeps its own (explicit) new position — it is excluded
/// from every other root's propagation, so the deltas never stack.
pub fn apply_group_moves(state: &mut EditorState, before: &[(HierRef, (f32, f32, f32))]) {
    // Which roots moved, and by how much.
    let mut moved: Vec<(HierRef, (f32, f32, f32))> = Vec::new();
    for &(r, (bx, by, bz)) in before {
        if let Some((ax, ay, az)) = position_of(state, r) {
            let d = (ax - bx, ay - by, az - bz);
            if d.0 != 0.0 || d.1 != 0.0 || d.2 != 0.0 {
                moved.push((r, d));
            }
        }
    }
    if moved.is_empty() {
        return;
    }
    let moved_set: Vec<HierRef> = moved.iter().map(|(r, _)| *r).collect();
    // Apply each root's delta to its descendants, skipping any descendant that is
    // itself a moved root (its explicit move already placed it).
    for (root, delta) in &moved {
        let descendants = state.level.hier_descendants(*root);
        for d in descendants {
            if moved_set.contains(&d) {
                continue;
            }
            translate(state, d, *delta);
        }
    }
}

// ───────────────────────────── reparent / unparent ─────────────────────────────

/// REPARENT key (`p`) in the tree. Two-step:
/// 1. First press on an object row BEGINS a reparent — remembers the child and
///    prompts the user to navigate to the intended parent.
/// 2. Second press CONFIRMS: the row the cursor is now on becomes the parent.
///    Refuses a self-parent or a cycle (status hint), leaving the map unchanged.
///
/// A `[hierarchy]` edit snapshots undo (it mutates the level). The child stays
/// selected. Pressing `p` on a non-object row, or while pending with the cursor on
/// a non-object row, reports it and (for the latter) keeps the pending pick.
pub fn reparent_key(state: &mut EditorState) {
    let here = current_object_target(state).map(target_to_ref);
    if here.is_none() {
        // No object under the cursor.
        if state.reparent_pending.is_some() {
            state.status = "reparent: move to a target object then press p".into();
        } else {
            state.status = "reparent: select an object first".into();
        }
        return;
    }
    match (state.reparent_pending, here) {
        (None, Some(child)) => {
            state.reparent_pending = Some(child);
            state.status = format!(
                "reparent {}: navigate to the PARENT, press p (or u to cancel)",
                ref_label(state, child)
            );
        }
        (Some(child), Some(parent)) => {
            state.reparent_pending = None;
            if state.level.hier_would_cycle(child, parent) {
                state.status = format!(
                    "reparent refused: {} cannot parent under {} (cycle/self)",
                    ref_label(state, child),
                    ref_label(state, parent)
                );
                return;
            }
            state.push_undo();
            state.level.set_hier_parent(child, parent);
            state.mark_dirty();
            state.status = format!(
                "{} parented under {}",
                ref_label(state, child),
                ref_label(state, parent)
            );
        }
        _ => {}
    }
}

/// UN-PARENT key (`u`) in the tree: clear the parent of the object the cursor is
/// on (or cancel a pending reparent). Snapshots undo only when a link is removed.
pub fn unparent_key(state: &mut EditorState) {
    if state.reparent_pending.take().is_some() {
        state.status = "reparent cancelled".into();
        return;
    }
    let Some(child) = current_object_target(state).map(target_to_ref) else {
        state.status = "unparent: select an object first".into();
        return;
    };
    if state.level.hier_parent_of(child).is_none() {
        state.status = format!("{} has no parent", ref_label(state, child));
        return;
    }
    state.push_undo();
    state.level.clear_hier_parent(child);
    state.mark_dirty();
    state.status = format!("{} un-parented (now a root)", ref_label(state, child));
}

/// The [`TreeTarget`] under the tree cursor, IF it is an object row (not a group
/// header / summary). Shared by the reparent/unparent keys.
fn current_object_target(state: &EditorState) -> Option<TreeTarget> {
    match crate::tree::current_row(state)? {
        // Pipes / ZoneInputs / Notes / WallDecals / Parts are not part of the
        // object-nesting model — never reparentable.
        crate::tree::TreeRow::Object {
            target:
                TreeTarget::Pipe(_)
                | TreeTarget::ZoneInput(_)
                | TreeTarget::Note(_)
                | TreeTarget::WallDecal(_)
                | TreeTarget::Holo(_)
                | TreeTarget::Portal(_)
                | TreeTarget::Zone(_)
                | TreeTarget::Line(_)
                | TreeTarget::Inset(_)
                | TreeTarget::Part(_),
            ..
        } => None,
        crate::tree::TreeRow::Object { target, .. } => Some(target),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_level::{HierKind, HierRef};

    /// A populated editor state: 3 nodes + an emitter, in a roomy grid.
    fn state_3nodes() -> EditorState {
        let mut s = EditorState::new_empty(32, 32, 8);
        s.level.push_node(2, 2, 1); // node 0
        s.level.push_node(5, 5, 1); // node 1
        s.level.push_node(8, 8, 1); // node 2
        s
    }

    #[test]
    fn target_ref_round_trip() {
        for t in [
            TreeTarget::Node(3),
            TreeTarget::Emitter(1),
            TreeTarget::Trigger(0),
            TreeTarget::Vox(2),
            TreeTarget::Singleton(EntityKind::Door),
            TreeTarget::Singleton(EntityKind::Spawn),
            TreeTarget::Singleton(EntityKind::Pedestal),
        ] {
            assert_eq!(ref_to_target(target_to_ref(t)), t, "round-trip {t:?}");
        }
    }

    #[test]
    fn group_move_drags_direct_child() {
        let mut s = state_3nodes();
        let a = HierRef::new(HierKind::Node, 0);
        let b = HierRef::new(HierKind::Node, 1);
        assert!(s.level.set_hier_parent(b, a));
        // Move parent a by (+3,+1,0): snapshot, mutate a, then apply group-move.
        let before = snapshot_positions(&s);
        s.level.nodes[0].x += 3;
        s.level.nodes[0].y += 1;
        apply_group_moves(&mut s, &before);
        // Child b moved by the SAME delta.
        assert_eq!((s.level.nodes[1].x, s.level.nodes[1].y), (8, 6));
    }

    #[test]
    fn group_move_is_multi_level() {
        let mut s = state_3nodes();
        let a = HierRef::new(HierKind::Node, 0);
        let b = HierRef::new(HierKind::Node, 1);
        let c = HierRef::new(HierKind::Node, 2);
        s.level.set_hier_parent(b, a); // b under a
        s.level.set_hier_parent(c, b); // c under b (grandchild of a)
        let (cx0, cy0) = (s.level.nodes[2].x, s.level.nodes[2].y);
        // Move grandparent a by (+10,+2,0).
        let before = snapshot_positions(&s);
        s.level.nodes[0].x += 10;
        s.level.nodes[0].y += 2;
        apply_group_moves(&mut s, &before);
        // The grandchild c moved by the SAME delta as the grandparent.
        assert_eq!(
            (s.level.nodes[2].x, s.level.nodes[2].y),
            (cx0 + 10, cy0 + 2)
        );
        // The middle child b also moved.
        assert_eq!((s.level.nodes[1].x, s.level.nodes[1].y), (15, 7));
    }

    #[test]
    fn group_move_no_links_is_noop() {
        let mut s = state_3nodes();
        let before = snapshot_positions(&s);
        assert!(before.is_empty(), "no links ⇒ empty snapshot");
        s.level.nodes[0].x += 5;
        apply_group_moves(&mut s, &before);
        // Siblings untouched.
        assert_eq!(s.level.nodes[1].x, 5);
    }

    #[test]
    fn reparent_and_unparent_via_keys() {
        let mut s = state_3nodes();
        s.tree.open = true;
        // Put the cursor on node 1's row and begin a reparent.
        let rows = crate::tree::build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    crate::tree::TreeRow::Object {
                        target: TreeTarget::Node(1),
                        ..
                    }
                )
            })
            .unwrap();
        reparent_key(&mut s); // begins (child = node 1)
        assert_eq!(s.reparent_pending, Some(HierRef::new(HierKind::Node, 1)));
        // Move cursor to node 0 and confirm → node1 parented under node0.
        let rows = crate::tree::build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    crate::tree::TreeRow::Object {
                        target: TreeTarget::Node(0),
                        ..
                    }
                )
            })
            .unwrap();
        reparent_key(&mut s);
        assert!(s.reparent_pending.is_none());
        assert_eq!(
            s.level.hier_parent_of(HierRef::new(HierKind::Node, 1)),
            Some(HierRef::new(HierKind::Node, 0))
        );
        // Unparent it again.
        let rows = crate::tree::build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    crate::tree::TreeRow::Object {
                        target: TreeTarget::Node(1),
                        ..
                    }
                )
            })
            .unwrap();
        unparent_key(&mut s);
        assert!(s
            .level
            .hier_parent_of(HierRef::new(HierKind::Node, 1))
            .is_none());
    }

    #[test]
    fn reparent_cycle_is_refused() {
        let mut s = state_3nodes();
        s.tree.open = true;
        let a = HierRef::new(HierKind::Node, 0);
        let b = HierRef::new(HierKind::Node, 1);
        s.level.set_hier_parent(b, a); // b under a
                                       // Now try to parent a under b (a cycle) via the keys.
        let rows = crate::tree::build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    crate::tree::TreeRow::Object {
                        target: TreeTarget::Node(0),
                        ..
                    }
                )
            })
            .unwrap();
        reparent_key(&mut s); // child = a
        let rows = crate::tree::build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    crate::tree::TreeRow::Object {
                        target: TreeTarget::Node(1),
                        ..
                    }
                )
            })
            .unwrap();
        reparent_key(&mut s); // parent = b → cycle → refused
        assert!(
            s.level.hier_parent_of(a).is_none(),
            "cycle refused, no link added"
        );
        assert!(s.status.contains("refused"));
    }
}
