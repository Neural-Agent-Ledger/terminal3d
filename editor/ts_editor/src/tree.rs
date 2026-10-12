//! OBJECT TREE / OUTLINER + object-TYPE SELECTOR palette + jump-to-object.
//!
//! Pure functions over [`EditorState`] — no I/O, no terminal. The outliner is a
//! READ VIEW over `level.*`: [`build_tree`] enumerates every object each frame
//! from the live level (no parallel state to desync). The only persistent state
//! is the small navigation cursor ([`EditorState::tree`]); the row list itself is
//! always rebuilt.
//!
//! Tree layout (one [`TreeRow`] per visible line):
//! ```text
//! Objects: 12                       (summary header — TreeRow::Summary)
//! v Nodes (5)                       (group header — collapsible, TreeRow::Group)
//!     n0 "in"   (1,5,2) input       (object — TreeRow::Object)
//!     n1 "out"  (10,3,1) output
//! > Emitters (3)                    (collapsed group; children hidden)
//! v Doors (1)
//!     door (11,3,1) on:2 off:0
//! ...
//! ```
//!
//! Selecting a row (Enter): sets the object selector to that object's KIND +
//! index (so the existing focused-field editors immediately edit it), jumps the
//! 2D cursor to its position, and moves the preview camera to look at it. The
//! structure (a flat list carrying an optional `depth`) is built so a future
//! parent/child hierarchy can render by indenting deeper rows — that NESTING is
//! the next milestone and is intentionally NOT implemented here.
use crate::state::{EditorState, EntityKind};

/// The object groups the outliner buckets every level object into, in display
/// order. `Vox` is its own group even though it is NOT an [`EntityKind`] (it
/// lives in vox-mode), so the tree can surface and select stamps too.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TreeGroup {
    Nodes,
    Pipes,
    Doors,
    Spawn,
    Pedestal,
    Parts,
    Emitters,
    Triggers,
    ZoneInputs,
    Notes,
    WallDecals,
    Holos,
    Portals,
    Zones,
    Lines,
    Insets,
    Vox,
}

/// Every group, in display order. Driving the build + the collapse bitset off
/// this one array keeps the order and the `collapsed` indexing in lock-step.
pub const TREE_GROUPS: [TreeGroup; 17] = [
    TreeGroup::Nodes,
    TreeGroup::Pipes,
    TreeGroup::Doors,
    TreeGroup::Spawn,
    TreeGroup::Pedestal,
    TreeGroup::Parts,
    TreeGroup::Emitters,
    TreeGroup::Triggers,
    TreeGroup::ZoneInputs,
    TreeGroup::Notes,
    TreeGroup::WallDecals,
    TreeGroup::Holos,
    TreeGroup::Portals,
    TreeGroup::Zones,
    TreeGroup::Lines,
    TreeGroup::Insets,
    TreeGroup::Vox,
];

impl TreeGroup {
    /// Plural display label for the group header.
    pub fn label(self) -> &'static str {
        match self {
            TreeGroup::Nodes => "Nodes",
            TreeGroup::Pipes => "Pipes",
            TreeGroup::Doors => "Doors",
            TreeGroup::Spawn => "Spawn",
            TreeGroup::Pedestal => "Pedestal",
            TreeGroup::Parts => "Parts",
            TreeGroup::Emitters => "Emitters",
            TreeGroup::Triggers => "Triggers",
            TreeGroup::ZoneInputs => "ZoneInputs",
            TreeGroup::Notes => "Notes",
            TreeGroup::WallDecals => "Wall decals",
            TreeGroup::Holos => "Holograms",
            TreeGroup::Portals => "Portals",
            TreeGroup::Zones => "Zones",
            TreeGroup::Lines => "Lines",
            TreeGroup::Insets => "Insets",
            TreeGroup::Vox => "Vox",
        }
    }

    /// Index of this group within [`TREE_GROUPS`] (its collapse-bit position).
    pub fn index(self) -> usize {
        TREE_GROUPS.iter().position(|&g| g == self).unwrap_or(0)
    }
}

/// What an object row points at — the live object it selects/edits. Mirrors the
/// editor's selection model: an [`EntityKind`] singleton (no index), an indexed
/// `Node`/`Emitter`/`Trigger`, or a `Vox` stamp (vox-mode, not an EntityKind).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TreeTarget {
    /// A one-per-level singleton: Door / Spawn / Pedestal.
    Singleton(EntityKind),
    /// `level.nodes[idx]`.
    Node(usize),
    /// `level.emitters[idx]`.
    Emitter(usize),
    /// `level.triggers[idx]`.
    Trigger(usize),
    /// `level.pipes[idx]`.
    Pipe(usize),
    /// `level.zone_inputs[idx]`.
    ZoneInput(usize),
    /// `level.notes[idx]`.
    Note(usize),
    /// `level.decals[idx]`.
    WallDecal(usize),
    /// `level.holos[idx]`.
    Holo(usize),
    /// `level.portals[idx]`.
    Portal(usize),
    /// `level.overlay.zones[idx]` (Scalable Material Overlay zone).
    Zone(usize),
    /// `level.overlay.lines[idx]` (Scalable Material Overlay line).
    Line(usize),
    /// `level.overlay.insets[idx]` (Scalable Material Overlay inset).
    Inset(usize),
    /// `level.parts[idx]`.
    Part(usize),
    /// `level.voxes[idx]` (selected in vox-mode, not via the object selector).
    Vox(usize),
}

/// One rendered tree line. The flat list is what the panel paints and what
/// navigation steps over (group headers are skippable separately from objects).
#[derive(Clone, Debug)]
pub enum TreeRow {
    /// The count summary banner at the very top (`Objects: N`).
    Summary { total: usize },
    /// A collapsible group header (`v Nodes (5)` / `> Emitters (3)`).
    Group {
        group: TreeGroup,
        count: usize,
        collapsed: bool,
    },
    /// One object line under a group, carrying the text + the selection target.
    /// `depth` is the indent level (0 today; reserved for future child nesting).
    Object {
        target: TreeTarget,
        text: String,
        depth: u8,
    },
}

/// Persistent outliner navigation state (the only non-derived tree state). The
/// row list is rebuilt each frame; this just remembers where the cursor sits and
/// which groups are folded.
#[derive(Clone, Debug, Default)]
pub struct TreeState {
    /// `true` when the outliner panel is OPEN and FOCUSED (capturing nav keys).
    pub open: bool,
    /// Highlighted row index into the freshly-built [`build_tree`] list. Clamped
    /// on use so a level edit that shrinks the list never points out of range.
    pub cursor: usize,
    /// One collapse flag per [`TREE_GROUPS`] entry (true ⇒ children hidden).
    pub collapsed: [bool; TREE_GROUPS.len()],
    /// The live row FILTER: `Some(query)` while `/` filtering is active —
    /// [`build_tree`] then keeps only object rows whose text fuzzy-matches
    /// (`crate::keymap::score`) and prunes childless group headers. `None`
    /// (the default) leaves the tree untouched.
    pub filter: Option<String>,
}

/// Build the full outliner row list from the live level — a pure read view, so
/// it always reflects `level.*` with no parallel state. Group headers are always
/// emitted (even for empty groups) so the structure is stable and discoverable;
/// a collapsed group emits its header but none of its children.
///
/// NESTING: within each type group the objects are rendered in HIERARCHICAL
/// order — a same-group child appears directly under its parent, INDENTED by its
/// ancestor depth (`TreeRow::Object.depth`). A child whose parent lives in another
/// group still gets a non-zero `depth` (its true ancestor count), so the nesting
/// reads visibly even across groups. A root object sits at depth 0. The group
/// counts are unchanged (every object still belongs to exactly one type group),
/// so collapse/fold and the summary stay consistent.
pub fn build_tree(state: &EditorState) -> Vec<TreeRow> {
    let collapsed = &state.tree.collapsed;
    let total = object_count(state);
    let mut rows = vec![TreeRow::Summary { total }];

    for &group in &TREE_GROUPS {
        let count = group_count(state, group);
        let folded = collapsed[group.index()];
        rows.push(TreeRow::Group {
            group,
            count,
            collapsed: folded,
        });
        if folded {
            continue;
        }
        // The targets in this group, in hierarchical order (a same-group child
        // immediately after its same-group parent).
        for target in group_targets_ordered(state, group) {
            let depth = hier_depth(state, target);
            let text = object_text(state, target);
            rows.push(TreeRow::Object {
                target,
                text,
                depth,
            });
        }
    }
    // ── LIVE FILTER ── applied HERE (the one row source) so the painter,
    // the mouse hit-test (`chrome::tree_row_at`), navigation, and the palette's
    // goto candidates all agree on the filtered view. Object rows survive when
    // their text fuzzy-matches; group headers keep only when a child survived.
    if let Some(query) = state.tree.filter.as_deref() {
        if !query.is_empty() {
            let kept: Vec<TreeRow> = rows
                .into_iter()
                .filter(|r| match r {
                    TreeRow::Object { text, .. } => crate::keymap::score(query, text).is_some(),
                    _ => true,
                })
                .collect();
            let mut pruned: Vec<TreeRow> = Vec::with_capacity(kept.len());
            for (i, r) in kept.iter().enumerate() {
                if matches!(r, TreeRow::Group { .. })
                    && !matches!(kept.get(i + 1), Some(TreeRow::Object { .. }))
                {
                    continue; // childless header under this filter
                }
                pruned.push(r.clone());
            }
            return pruned;
        }
    }
    rows
}

/// The display text for one object [`TreeTarget`] (the per-kind readout). Pure.
fn object_text(state: &EditorState, target: TreeTarget) -> String {
    let lvl = &state.level;
    match target {
        TreeTarget::Node(i) => {
            let n = &lvl.nodes[i];
            let role = crate::entity::io_role_label(n.is_input, n.is_output);
            format!("{} \"{}\" ({},{},{}) {}", i, n.name, n.x, n.y, n.z, role)
        }
        TreeTarget::Emitter(i) => {
            let e = &lvl.emitters[i];
            let (ex, ey, ez) = emitter_cell_or_screen(e);
            format!("#{} {} ({},{},{}) r{:.0}", i, e.family, ex, ey, ez, e.rate)
        }
        TreeTarget::Trigger(i) => {
            let t = &lvl.triggers[i];
            let (min, _max) = trigger_min_max(t);
            let action = crate::entity::trigger_action_label(state, t);
            format!(
                "#{} ({},{},{}) {} once:{}",
                i, min.0, min.1, min.2, action, t.once
            )
        }
        TreeTarget::Pipe(i) => {
            let p = &lvl.pipes[i];
            let st = if p.state == re_level::PipeState::On {
                "on"
            } else {
                "off"
            };
            let sw = if p.switch_cell.is_some() {
                " switch"
            } else {
                ""
            };
            format!("#{} {}->{} {}{}", i, p.from, p.to, st, sw)
        }
        TreeTarget::ZoneInput(i) => {
            let zi = &lvl.zone_inputs[i];
            let node = if zi.node.is_empty() {
                "(unlinked)"
            } else {
                zi.node.as_str()
            };
            format!("#{} ({},{},{}) -> {}", i, zi.x0, zi.y0, zi.z0, node)
        }
        TreeTarget::Note(i) => {
            let n = &lvl.notes[i];
            let body: String = n.text.chars().take(16).collect();
            format!(
                "#{} {} ({},{},{}) \"{}\"",
                i,
                crate::state::note_kind_label(n.kind),
                n.x,
                n.y,
                n.z,
                body
            )
        }
        TreeTarget::WallDecal(i) => {
            let d = &lvl.decals[i];
            format!(
                "#{} {} ({},{},{}) \"{}\"",
                i,
                crate::state::decal_surface_label(d.surface_kind),
                d.x0,
                d.y0,
                d.z0,
                d.name
            )
        }
        TreeTarget::Holo(i) => {
            let h = &lvl.holos[i];
            format!(
                "#{} {} ({},{},{}) \"{}\"",
                i,
                crate::state::holo_shape_label(h.shape),
                h.x0,
                h.y0,
                h.z0,
                h.name
            )
        }
        TreeTarget::Portal(i) => {
            let p = &lvl.portals[i];
            format!(
                "#{} A({},{},{}) B({},{},{}) \"{}\"",
                i, p.ax0, p.ay0, p.az0, p.bx0, p.by0, p.bz0, p.name
            )
        }
        TreeTarget::Zone(i) => {
            let z = &lvl.overlay.zones[i];
            let (min, _max) = match z.region {
                re_core::Region::Box { min, max } => (min, max),
                _ => ((0, 0, 0), (0, 0, 0)),
            };
            format!(
                "#{} \"{}\" ({},{},{}) {}",
                i,
                z.name,
                min.0,
                min.1,
                min.2,
                crate::state::scale_rule_label(&z.scale)
            )
        }
        TreeTarget::Line(i) => {
            let l = &lvl.overlay.lines[i];
            format!(
                "#{} \"{}\" {} pts {}",
                i,
                l.name,
                l.path.points.len(),
                crate::state::scale_rule_label(&l.scale)
            )
        }
        TreeTarget::Inset(i) => {
            let ins = &lvl.overlay.insets[i];
            let (min, _max) = match ins.region {
                re_core::Region::Box { min, max } => (min, max),
                _ => ((0, 0, 0), (0, 0, 0)),
            };
            format!(
                "#{} \"{}\" oct {}-{} ({},{},{})",
                i, ins.name, ins.at_octave.0, ins.at_octave.1, min.0, min.1, min.2
            )
        }
        TreeTarget::Part(i) => {
            let p = &lvl.parts[i];
            format!(
                "#{} \"{}\" {}x{}x{} @({},{},{})",
                i, p.name, p.w, p.h, p.d, p.off_x, p.off_y, p.off_z
            )
        }
        TreeTarget::Vox(i) => {
            let v = &lvl.voxes[i];
            let file = short_file(&v.file);
            format!(
                "#{} {} ({},{},{}) x{}",
                i,
                file,
                v.off_x,
                v.off_y,
                v.off_z,
                v.scale.max(1)
            )
        }
        TreeTarget::Singleton(EntityKind::Door) => {
            let d = &lvl.door;
            format!(
                "door ({},{},{}) on:{} off:{}",
                d.x,
                d.y,
                d.z,
                d.target_parts.len(),
                d.target_off.len()
            )
        }
        TreeTarget::Singleton(EntityKind::Spawn) => {
            let p = &lvl.player;
            format!(
                "spawn ({},{},{}) {}",
                p.x,
                p.y,
                p.z,
                crate::state::facing_label(p.facing_yaw_col)
            )
        }
        TreeTarget::Singleton(EntityKind::Pedestal) => {
            let p = &lvl.pedestal;
            format!(
                "pedestal ({},{},{}) {}x{}x{}",
                p.cx,
                p.cy,
                p.cz,
                p.w.max(1),
                p.h.max(1),
                p.d.max(1)
            )
        }
        TreeTarget::Singleton(_) => String::new(),
    }
}

/// The raw (flat, index-order) list of targets a group holds, before hierarchical
/// re-ordering. Singletons emit one entry only when present.
fn group_targets_flat(state: &EditorState, group: TreeGroup) -> Vec<TreeTarget> {
    let lvl = &state.level;
    match group {
        TreeGroup::Nodes => (0..lvl.nodes.len()).map(TreeTarget::Node).collect(),
        TreeGroup::Pipes => (0..lvl.pipes.len()).map(TreeTarget::Pipe).collect(),
        // SYNTHESIZED emitters (e.g. the `[effects] snow` lattice) are not
        // editor objects — regenerated wholesale from a knob on every load and
        // never persisted by `to_lvl_text` — so they never appear in the tree.
        // See `EmitterSpec::synthesized` and `entity::emitter_at`.
        TreeGroup::Emitters => (0..lvl.emitters.len())
            .filter(|&i| !lvl.emitters[i].synthesized)
            .map(TreeTarget::Emitter)
            .collect(),
        TreeGroup::Triggers => (0..lvl.triggers.len()).map(TreeTarget::Trigger).collect(),
        TreeGroup::ZoneInputs => (0..lvl.zone_inputs.len())
            .map(TreeTarget::ZoneInput)
            .collect(),
        TreeGroup::Notes => (0..lvl.notes.len()).map(TreeTarget::Note).collect(),
        TreeGroup::WallDecals => (0..lvl.decals.len()).map(TreeTarget::WallDecal).collect(),
        TreeGroup::Holos => (0..lvl.holos.len()).map(TreeTarget::Holo).collect(),
        TreeGroup::Portals => (0..lvl.portals.len()).map(TreeTarget::Portal).collect(),
        TreeGroup::Zones => (0..lvl.overlay.zones.len()).map(TreeTarget::Zone).collect(),
        TreeGroup::Lines => (0..lvl.overlay.lines.len()).map(TreeTarget::Line).collect(),
        TreeGroup::Insets => (0..lvl.overlay.insets.len())
            .map(TreeTarget::Inset)
            .collect(),
        TreeGroup::Parts => (0..lvl.parts.len()).map(TreeTarget::Part).collect(),
        TreeGroup::Vox => (0..lvl.voxes.len()).map(TreeTarget::Vox).collect(),
        TreeGroup::Doors => present_singleton(state, EntityKind::Door),
        TreeGroup::Spawn => present_singleton(state, EntityKind::Spawn),
        TreeGroup::Pedestal => present_singleton(state, EntityKind::Pedestal),
    }
}

/// A one-element (present) or empty (absent) singleton target list.
fn present_singleton(state: &EditorState, kind: EntityKind) -> Vec<TreeTarget> {
    let present = match kind {
        EntityKind::Door => door_present(state),
        EntityKind::Spawn => spawn_present(state),
        EntityKind::Pedestal => pedestal_present(state),
        _ => false,
    };
    if present {
        vec![TreeTarget::Singleton(kind)]
    } else {
        Vec::new()
    }
}

/// The group's targets in HIERARCHICAL order: a same-group child is emitted
/// directly after its same-group parent (depth-first), so nesting within a group
/// reads top-down. A target whose parent is OUTSIDE this group is treated as a
/// local root (it still carries its true cross-group `depth`, computed separately
/// in [`hier_depth`]). Roots keep their original index order; children follow.
fn group_targets_ordered(state: &EditorState, group: TreeGroup) -> Vec<TreeTarget> {
    let flat = group_targets_flat(state, group);
    if flat.len() <= 1 {
        return flat;
    }
    // Pipes / ZoneInputs / Notes are not part of the object-NESTING hierarchy (no
    // matching `HierKind`), so they never reparent — keep their natural index order
    // without consulting the hierarchy (which would mis-map their index onto a node
    // ref).
    if matches!(
        group,
        TreeGroup::Pipes
            | TreeGroup::ZoneInputs
            | TreeGroup::Notes
            | TreeGroup::WallDecals
            | TreeGroup::Holos
            | TreeGroup::Portals
            | TreeGroup::Zones
            | TreeGroup::Lines
            | TreeGroup::Insets
            | TreeGroup::Parts
    ) {
        return flat;
    }
    // Map each target to its HierRef + parent.
    let in_group: Vec<(TreeTarget, re_level::HierRef)> = flat
        .iter()
        .map(|&t| (t, crate::hierarchy::target_to_ref(t)))
        .collect();
    let refs_in_group: Vec<re_level::HierRef> = in_group.iter().map(|(_, r)| *r).collect();
    // Local roots: targets whose parent is NOT another target in THIS group.
    let mut roots: Vec<TreeTarget> = Vec::new();
    for &(t, r) in &in_group {
        let parent_in_group = state
            .level
            .hier_parent_of(r)
            .is_some_and(|p| refs_in_group.contains(&p));
        if !parent_in_group {
            roots.push(t);
        }
    }
    // Depth-first emit each root, then its same-group descendants under it.
    let mut out: Vec<TreeTarget> = Vec::new();
    for root in roots {
        emit_subtree(state, root, &in_group, &mut out);
    }
    // Safety net: if a malformed map left some target unvisited (shouldn't happen
    // after pruning), append the leftovers in index order so nothing vanishes.
    for &(t, _) in &in_group {
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// Depth-first append `target` then its same-group children (recursively) to `out`.
fn emit_subtree(
    state: &EditorState,
    target: TreeTarget,
    in_group: &[(TreeTarget, re_level::HierRef)],
    out: &mut Vec<TreeTarget>,
) {
    if out.contains(&target) {
        return; // cycle / re-entry guard
    }
    out.push(target);
    let r = crate::hierarchy::target_to_ref(target);
    for &(child_t, child_r) in in_group {
        if state.level.hier_parent_of(child_r) == Some(r) {
            emit_subtree(state, child_t, in_group, out);
        }
    }
}

/// The ancestor depth of a target across ALL kinds (number of parents up to a
/// root). 0 ⇒ a root object. Drives the row indent so nesting is visible even
/// when parent and child sit in different type groups. Bounded by the link count.
fn hier_depth(state: &EditorState, target: TreeTarget) -> u8 {
    // Pipes / ZoneInputs / Notes / WallDecals / Parts never nest (no matching
    // `HierKind`); always a root.
    if matches!(
        target,
        TreeTarget::Pipe(_)
            | TreeTarget::ZoneInput(_)
            | TreeTarget::Note(_)
            | TreeTarget::WallDecal(_)
            | TreeTarget::Holo(_)
            | TreeTarget::Portal(_)
            | TreeTarget::Zone(_)
            | TreeTarget::Line(_)
            | TreeTarget::Inset(_)
            | TreeTarget::Part(_)
    ) {
        return 0;
    }
    let mut depth: u32 = 0;
    let mut cur = crate::hierarchy::target_to_ref(target);
    let max = state.level.hierarchy.len() as u32;
    while let Some(p) = state.level.hier_parent_of(cur) {
        depth += 1;
        cur = p;
        if depth > max {
            break; // cycle guard (should never fire after pruning)
        }
    }
    depth.min(u8::MAX as u32) as u8
}

/// Total authored objects across every group (the summary count).
pub fn object_count(state: &EditorState) -> usize {
    TREE_GROUPS.iter().map(|&g| group_count(state, g)).sum()
}

/// How many objects a group holds (0 or 1 for the singletons).
fn group_count(state: &EditorState, group: TreeGroup) -> usize {
    let lvl = &state.level;
    match group {
        TreeGroup::Nodes => lvl.nodes.len(),
        TreeGroup::Pipes => lvl.pipes.len(),
        TreeGroup::Doors => usize::from(door_present(state)),
        TreeGroup::Spawn => usize::from(spawn_present(state)),
        TreeGroup::Pedestal => usize::from(pedestal_present(state)),
        TreeGroup::Parts => lvl.parts.len(),
        // Mirrors `group_targets_flat`: synthesized emitters are excluded.
        TreeGroup::Emitters => lvl.emitters.iter().filter(|e| !e.synthesized).count(),
        TreeGroup::Triggers => lvl.triggers.len(),
        TreeGroup::ZoneInputs => lvl.zone_inputs.len(),
        TreeGroup::Notes => lvl.notes.len(),
        TreeGroup::WallDecals => lvl.decals.len(),
        TreeGroup::Holos => lvl.holos.len(),
        TreeGroup::Portals => lvl.portals.len(),
        TreeGroup::Zones => lvl.overlay.zones.len(),
        TreeGroup::Lines => lvl.overlay.lines.len(),
        TreeGroup::Insets => lvl.overlay.insets.len(),
        TreeGroup::Vox => lvl.voxes.len(),
    }
}

/// A door is "present" when its anchor is not the unset origin (matches the
/// top-down marker's visibility rule so the tree and the grid agree).
fn door_present(state: &EditorState) -> bool {
    let d = &state.level.door;
    (d.x, d.y, d.z) != (0, 0, 0)
}

/// A spawn is "present" when its pose is not the unset origin.
fn spawn_present(state: &EditorState) -> bool {
    let p = &state.level.player;
    (p.x, p.y, p.z) != (0, 0, 0)
}

/// A pedestal is "present" when it has any extent (an all-zero pedestal is unset).
fn pedestal_present(state: &EditorState) -> bool {
    let p = &state.level.pedestal;
    p.w != 0 || p.h != 0 || p.d != 0
}

/// The world cell an emitter sits at (floored origin), or its screen anchor for a
/// `Screen` origin — just for the readout.
fn emitter_cell_or_screen(e: &re_world::EmitterSpec) -> (i32, i32, i32) {
    match e.origin {
        re_world::EmitterOrigin::World(p) => {
            (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
        }
        re_world::EmitterOrigin::Screen { col, row } => (col as i32, row as i32, 0),
    }
}

/// A trigger's box corners (min,max). A non-Box region reports an origin point.
fn trigger_min_max(t: &re_core::Trigger) -> ((i32, i32, i32), (i32, i32, i32)) {
    match t.region {
        re_core::Region::Box { min, max } => (min, max),
        _ => ((0, 0, 0), (0, 0, 0)),
    }
}

/// Trim a vox file path to its file name for the compact readout.
fn short_file(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

// ───────────────────────────── navigation ─────────────────────────────

/// Toggle the outliner panel open/closed. Opening focuses it (captures nav keys)
/// and clamps the cursor to a valid row; closing returns the arrows to the 2D
/// cursor. Mutually exclusive with the type-selector palette.
pub fn toggle(state: &mut EditorState) {
    state.tree.open = !state.tree.open;
    // Closing the tree (or re-opening it) abandons any in-flight reparent pick.
    state.reparent_pending = None;
    if state.tree.open {
        state.palette = None;
        clamp_cursor(state);
        state.status =
            "tree (Up/Dn, Enter sel, h fold, z dup, d del, p parent, u unparent, t close)".into();
    } else {
        state.status = "object tree closed".into();
    }
}

/// Clamp the tree cursor into the current row list (a level edit may have shrunk
/// it). A tree with only the summary row keeps the cursor at 0.
pub fn clamp_cursor(state: &mut EditorState) {
    let n = build_tree(state).len();
    if n == 0 {
        state.tree.cursor = 0;
    } else {
        state.tree.cursor = state.tree.cursor.min(n - 1);
    }
}

/// Move the highlighted row by `delta` (±1), clamped to the list ends (no wrap so
/// the summary banner at the top and the last object stay reachable as anchors).
pub fn navigate(state: &mut EditorState, delta: i32) {
    let rows = build_tree(state);
    if rows.is_empty() {
        state.tree.cursor = 0;
        return;
    }
    let last = rows.len() as i32 - 1;
    let next = (state.tree.cursor as i32 + delta).clamp(0, last);
    state.tree.cursor = next as usize;
    // Echo the landed row so the status line tracks the selection.
    state.status = match &rows[state.tree.cursor] {
        TreeRow::Summary { total } => format!("tree: Objects {total}"),
        TreeRow::Group {
            group,
            count,
            collapsed,
        } => {
            let mark = if *collapsed { "folded" } else { "open" };
            format!("tree: {} ({count}) [{mark}]", group.label())
        }
        TreeRow::Object { text, .. } => format!("tree: {text}"),
    };
}

/// The row the tree cursor currently sits on (clamped), or `None` for an empty
/// list. Used by the panel painter and the select/fold/dup/del routes.
pub fn current_row(state: &EditorState) -> Option<TreeRow> {
    let rows = build_tree(state);
    rows.get(state.tree.cursor.min(rows.len().saturating_sub(1)))
        .cloned()
}

/// Collapse/expand the group the cursor is on — works whether the cursor is on
/// the group HEADER or on one of its OBJECT rows (so `h` always folds the
/// enclosing group). A no-op (with a hint) on the summary banner.
pub fn toggle_collapse(state: &mut EditorState) {
    let Some(group) = enclosing_group(state) else {
        state.status = "tree: nothing to fold here".into();
        return;
    };
    let i = group.index();
    state.tree.collapsed[i] = !state.tree.collapsed[i];
    // Re-clamp: folding a group shortens the list and may strand the cursor.
    clamp_cursor(state);
    let mark = if state.tree.collapsed[i] {
        "folded"
    } else {
        "open"
    };
    state.status = format!("tree: {} {mark}", group.label());
}

/// The group enclosing the cursor row: the group itself when on a header, or the
/// owning group when on an object row. `None` on the summary banner.
fn enclosing_group(state: &EditorState) -> Option<TreeGroup> {
    let rows = build_tree(state);
    let idx = state.tree.cursor.min(rows.len().saturating_sub(1));
    // Walk UP from the cursor to the nearest group header.
    for row in rows[..=idx].iter().rev() {
        if let TreeRow::Group { group, .. } = row {
            return Some(*group);
        }
    }
    None
}

// ───────────────────────────── select / jump ─────────────────────────────

/// SELECT + JUMP-TO the object on the cursor row: point the editor's object
/// selector at it (so the existing `;`/`'`/`+`/`-`/`c` field editors edit it),
/// jump the 2D cursor to its position, and move the preview camera to look at it.
/// A group header toggles its fold; the summary banner is inert.
pub fn select_current(state: &mut EditorState) {
    let Some(row) = current_row(state) else {
        return;
    };
    match row {
        TreeRow::Summary { .. } => {
            state.status = "tree: summary (select an object below)".into();
        }
        TreeRow::Group { .. } => toggle_collapse(state),
        // While a PIPE-CREATE is pending, Enter on a NODE row picks it as the
        // source/target (the connect flow), rather than a plain select.
        TreeRow::Object {
            target: TreeTarget::Node(i),
            ..
        } if state.pipe_pending.is_some() => {
            crate::pipe::pick_node(state, i);
        }
        TreeRow::Object { target, .. } => select_target(state, target),
    }
}

/// Point the editor's selection at `target`, move the 2D cursor to its cell, and
/// jump the preview camera there. Shared by Enter-select and post-duplicate
/// re-selection. Leaves the tree open (so you can keep browsing/editing).
pub fn select_target(state: &mut EditorState, target: TreeTarget) {
    // Selecting an object clears the OTHER selections so a stale node/emitter/
    // trigger edit can't leak across kinds (mirrors `cycle_selector`).
    state.selected_node = None;
    state.selected_emitter = None;
    state.selected_trigger = None;
    state.selected_vox = None;
    state.selected_pipe = None;
    state.selected_zone_input = None;
    state.selected_note = None;
    state.selected_decal = None;
    state.selected_holo = None;
    state.selected_portal = None;
    state.selected_zone = None;
    state.selected_line = None;
    state.selected_inset = None;
    state.selected_part = None;
    let (pos, label) = match target {
        TreeTarget::Singleton(kind) => {
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(kind);
            if let Some(f) = crate::state::entity_field_default(kind) {
                state.entity_field = f;
            }
            let p = singleton_pos(state, kind);
            (p, format!("selected {}", kind.label()))
        }
        TreeTarget::Node(i) => {
            let i = i.min(state.level.nodes.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Node);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Node).unwrap_or(state.entity_field);
            state.selected_node = state.level.nodes.get(i).map(|_| i);
            let p = state
                .level
                .nodes
                .get(i)
                .map(|n| (n.x, n.y, n.z))
                .unwrap_or((0, 0, 0));
            let name = state
                .level
                .nodes
                .get(i)
                .map(|n| n.name.clone())
                .unwrap_or_default();
            (p, format!("selected node '{name}'"))
        }
        TreeTarget::Emitter(i) => {
            let i = i.min(state.level.emitters.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Emitter);
            state.selected_emitter = state.level.emitters.get(i).map(|_| i);
            let p = state
                .level
                .emitters
                .get(i)
                .map(emitter_cell_or_screen)
                .unwrap_or((0, 0, 0));
            (p, format!("selected emitter #{i}"))
        }
        TreeTarget::Trigger(i) => {
            let i = i.min(state.level.triggers.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Trigger);
            state.entity_field = crate::state::entity_field_default(EntityKind::Trigger)
                .unwrap_or(state.entity_field);
            state.selected_trigger = state.level.triggers.get(i).map(|_| i);
            let p = state
                .level
                .triggers
                .get(i)
                .map(|t| trigger_min_max(t).0)
                .unwrap_or((0, 0, 0));
            (p, format!("selected trigger #{i}"))
        }
        TreeTarget::Pipe(i) => {
            let i = i.min(state.level.pipes.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Pipe);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Pipe).unwrap_or(state.entity_field);
            state.selected_pipe = state.level.pipes.get(i).map(|_| i);
            // Jump to the SWITCH cell if one is set, else the source-node cell.
            let p = state.level.pipes.get(i).map(|p| {
                p.switch_cell
                    .and_then(|c| p.path.get(c).copied())
                    .or_else(|| p.path.first().copied())
                    .unwrap_or((0, 0, 0))
            });
            (p.unwrap_or((0, 0, 0)), format!("selected pipe #{i}"))
        }
        TreeTarget::ZoneInput(i) => {
            let i = i.min(state.level.zone_inputs.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::ZoneInput);
            state.entity_field = crate::state::entity_field_default(EntityKind::ZoneInput)
                .unwrap_or(state.entity_field);
            state.selected_zone_input = state.level.zone_inputs.get(i).map(|_| i);
            // Sync the input-node cursor to the zone's current link so the ZoneNode
            // field opens "where the link already is".
            if let Some(zi) = state.level.zone_inputs.get(i) {
                state.zone_node_cursor = zi.node_index;
            }
            let p = state
                .level
                .zone_inputs
                .get(i)
                .map(|zi| (zi.x0, zi.y0, zi.z0))
                .unwrap_or((0, 0, 0));
            (p, format!("selected zone input #{i}"))
        }
        TreeTarget::Note(i) => {
            let i = i.min(state.level.notes.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Note);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Note).unwrap_or(state.entity_field);
            state.selected_note = state.level.notes.get(i).map(|_| i);
            let p = state
                .level
                .notes
                .get(i)
                .map(|n| (n.x, n.y, n.z))
                .unwrap_or((0, 0, 0));
            (p, format!("selected note #{i}"))
        }
        TreeTarget::WallDecal(i) => {
            let i = i.min(state.level.decals.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::WallDecal);
            state.entity_field = crate::state::entity_field_default(EntityKind::WallDecal)
                .unwrap_or(state.entity_field);
            state.selected_decal = state.level.decals.get(i).map(|_| i);
            let p = state
                .level
                .decals
                .get(i)
                .map(|d| (d.x0, d.y0, d.z0))
                .unwrap_or((0, 0, 0));
            (p, format!("selected wall decal #{i}"))
        }
        TreeTarget::Holo(i) => {
            let i = i.min(state.level.holos.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Holo);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Holo).unwrap_or(state.entity_field);
            state.selected_holo = state.level.holos.get(i).map(|_| i);
            let p = state
                .level
                .holos
                .get(i)
                .map(|h| (h.x0, h.y0, h.z0))
                .unwrap_or((0, 0, 0));
            (p, format!("selected hologram #{i}"))
        }
        TreeTarget::Portal(i) => {
            let i = i.min(state.level.portals.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Portal);
            state.entity_field = crate::state::entity_field_default(EntityKind::Portal)
                .unwrap_or(state.entity_field);
            state.selected_portal = state.level.portals.get(i).map(|_| i);
            // Jump to Region A's min corner so the cursor lands on the portal.
            let p = state
                .level
                .portals
                .get(i)
                .map(|p| (p.ax0, p.ay0, p.az0))
                .unwrap_or((0, 0, 0));
            (p, format!("selected portal #{i}"))
        }
        TreeTarget::Zone(i) => {
            let i = i.min(state.level.overlay.zones.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Zone);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Zone).unwrap_or(state.entity_field);
            state.selected_zone = state.level.overlay.zones.get(i).map(|_| i);
            let p = state
                .level
                .overlay
                .zones
                .get(i)
                .map(|z| match z.region {
                    re_core::Region::Box { min, .. } => min,
                    _ => (0, 0, 0),
                })
                .unwrap_or((0, 0, 0));
            (p, format!("selected zone #{i}"))
        }
        TreeTarget::Line(i) => {
            let i = i.min(state.level.overlay.lines.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Line);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Line).unwrap_or(state.entity_field);
            state.selected_line = state.level.overlay.lines.get(i).map(|_| i);
            // Jump to the FIRST waypoint's cell so the cursor lands on the line.
            let p = state
                .level
                .overlay
                .lines
                .get(i)
                .and_then(|l| l.path.points.first())
                .map(|pt| {
                    (
                        pt.x.floor() as i32,
                        pt.y.floor() as i32,
                        pt.z.floor() as i32,
                    )
                })
                .unwrap_or((0, 0, 0));
            (p, format!("selected line #{i}"))
        }
        TreeTarget::Inset(i) => {
            let i = i.min(state.level.overlay.insets.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Inset);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Inset).unwrap_or(state.entity_field);
            state.selected_inset = state.level.overlay.insets.get(i).map(|_| i);
            let p = state
                .level
                .overlay
                .insets
                .get(i)
                .map(|ins| match ins.region {
                    re_core::Region::Box { min, .. } => min,
                    _ => (0, 0, 0),
                })
                .unwrap_or((0, 0, 0));
            (p, format!("selected inset #{i}"))
        }
        TreeTarget::Part(i) => {
            let i = i.min(state.level.parts.len().saturating_sub(1));
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(EntityKind::Part);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Part).unwrap_or(state.entity_field);
            state.selected_part = state.level.parts.get(i).map(|_| i);
            state.emit_part_cursor = i;
            // Opening a part for edit resets the mask layer cursor into range.
            state.mask_layer = 0;
            state.mask_cursor = (0, 0);
            let p = state
                .level
                .parts
                .get(i)
                .map(|p| (p.off_x, p.off_y, p.off_z))
                .unwrap_or((0, 0, 0));
            let name = state
                .level
                .parts
                .get(i)
                .map(|p| p.name.clone())
                .unwrap_or_default();
            (p, format!("selected part '{name}'"))
        }
        TreeTarget::Vox(i) => {
            let i = i.min(state.level.voxes.len().saturating_sub(1));
            // Vox lives in vox-mode, not the object selector.
            state.place = None;
            state.material_mode = false;
            state.vox_mode = true;
            state.selected_vox = state.level.voxes.get(i).map(|_| i);
            let p = state
                .level
                .voxes
                .get(i)
                .map(|v| (v.off_x, v.off_y, v.off_z))
                .unwrap_or((0, 0, 0));
            (p, format!("selected vox #{i}"))
        }
    };
    move_cursor_to(state, pos);
    crate::camera::jump_to_cursor(state);
    state.status = label;
}

/// The cell position of a singleton kind (Door / Spawn / Pedestal). Other kinds
/// have no singleton position (returns the origin).
fn singleton_pos(state: &EditorState, kind: EntityKind) -> (i32, i32, i32) {
    let lvl = &state.level;
    match kind {
        EntityKind::Door => (lvl.door.x, lvl.door.y, lvl.door.z),
        EntityKind::Spawn => (lvl.player.x, lvl.player.y, lvl.player.z),
        EntityKind::Pedestal => (lvl.pedestal.cx, lvl.pedestal.cy, lvl.pedestal.cz),
        _ => (0, 0, 0),
    }
}

/// Move the 2D cursor + active z to a world cell, clamped into the map bounds so
/// a stale/out-of-room coordinate never strands the cursor off-grid.
fn move_cursor_to(state: &mut EditorState, (x, y, z): (i32, i32, i32)) {
    let (sx, sy, sz) = state.map.size();
    let cx = x.clamp(0, sx.saturating_sub(1) as i32) as usize;
    let cy = y.clamp(0, sy.saturating_sub(1) as i32) as usize;
    let cz = z.clamp(0, sz.saturating_sub(1) as i32) as usize;
    state.cursor = (cx, cy);
    state.z = cz;
}

// ───────────────────────── convenience: delete / duplicate ─────────────────────────

/// DELETE the object the tree cursor is on, routing to the SAME delete logic the
/// `x` key uses (so behaviour can't diverge): it positions the editor's selector
/// and cursor on the object, then calls the existing `entity` / `vox` deleters.
/// Singletons (Door/Spawn/Pedestal) clear back to default; indexed objects are
/// removed. A group/summary row is a defined no-op (with a hint). Snapshots undo
/// before the change.
pub fn delete_current(state: &mut EditorState) {
    let Some(TreeRow::Object { target, .. }) = current_row(state) else {
        state.status = "tree: select an object to delete".into();
        return;
    };
    state.push_undo();
    // Point the selector + cursor at the object, then reuse the existing deleter
    // (which keys off `place` + the cursor cell). This shares one delete path.
    select_target(state, target);
    // The deleters (`entity::delete_at_cursor` / `vox::delete_at_cursor`) own the
    // hierarchy INDEX FIXUP after the `Vec::remove` (keyed off the real removed
    // index), so the direct `x` delete and the tree delete stay consistent.
    match target {
        TreeTarget::Vox(_) => crate::vox::delete_at_cursor(state),
        _ => crate::entity::delete_at_cursor(state),
    }
    clamp_cursor(state);
}

/// DUPLICATE the object the tree cursor is on at a +1 offset, then SELECT the
/// copy (so a follow-up edit targets it). Only the multi-instance kinds
/// (Node / Emitter / Trigger / Vox) duplicate — the singletons (Door / Spawn /
/// Pedestal) are one-per-level and report that. Snapshots undo.
pub fn duplicate_current(state: &mut EditorState) {
    let Some(TreeRow::Object { target, .. }) = current_row(state) else {
        state.status = "tree: select an object to duplicate".into();
        return;
    };
    let (sx, sy, sz) = state.map.size();
    // Offset the copy onto a DIFFERENT cell than the source. Prefer +1 toward
    // the next cell; at the far edge of the axis there is no +1 cell, so step
    // back to v-1 instead (else the copy would land on top of the original and
    // be invisible). Only when the axis is a single cell wide is collapse
    // unavoidable — then we keep v.
    let off = |v: i32, ext: usize| {
        let last = ext.saturating_sub(1) as i32;
        if v < last {
            v + 1
        } else if v > 0 {
            v - 1
        } else {
            v
        }
    };
    state.push_undo();
    let new_target = match target {
        TreeTarget::Node(i) => {
            let Some(src) = state.level.nodes.get(i).cloned() else {
                state.status = "tree: node gone".into();
                return;
            };
            let (nx, ny, nz) = (off(src.x, sx), off(src.y, sy), off(src.z, sz));
            // Reuse push_node for a guaranteed-unique name, then copy the params.
            let idx = state.level.push_node(nx, ny, nz);
            let name = state.level.nodes[idx].name.clone();
            let n = &mut state.level.nodes[idx];
            n.glyph = src.glyph;
            n.pool = src.pool.clone();
            n.pool_alt = src.pool_alt.clone();
            n.anchor = src.anchor;
            n.color_rgb = src.color_rgb;
            n.is_input = src.is_input;
            n.is_output = src.is_output;
            n.threshold = src.threshold;
            state.status = format!("duplicated node -> '{name}'");
            Some(TreeTarget::Node(idx))
        }
        TreeTarget::Emitter(i) => {
            let Some(mut spec) = state.level.emitters.get(i).cloned() else {
                state.status = "tree: emitter gone".into();
                return;
            };
            // Nudge a World origin by +1 cell; a Screen origin is left as-is.
            // At the far +x edge there is no room ahead, so step back by 1 so the
            // copy never lands on top of the source (matching the cell `off`).
            if let re_world::EmitterOrigin::World(p) = spec.origin {
                let max_x = sx.saturating_sub(1) as f32 + 0.5;
                let nx = if p.x + 1.0 <= max_x {
                    p.x + 1.0
                } else if p.x - 1.0 >= 0.0 {
                    p.x - 1.0
                } else {
                    p.x
                };
                spec.origin = re_world::EmitterOrigin::World(glam::Vec3::new(nx, p.y, p.z));
            }
            spec.validate();
            state.level.emitters.push(spec);
            let idx = state.level.emitters.len() - 1;
            state.status = format!("duplicated emitter -> #{idx}");
            Some(TreeTarget::Emitter(idx))
        }
        TreeTarget::Trigger(i) => {
            let Some(src) = state.level.triggers.get(i).cloned() else {
                state.status = "tree: trigger gone".into();
                return;
            };
            let (min, max) = trigger_min_max(&src);
            let shift = |c: (i32, i32, i32)| (off(c.0, sx), c.1, c.2);
            let mut copy = src.clone();
            copy.region = re_core::Region::Box {
                min: shift(min),
                max: shift(max),
            };
            state.level.triggers.push(copy);
            let idx = state.level.triggers.len() - 1;
            state.status = format!("duplicated trigger -> #{idx}");
            Some(TreeTarget::Trigger(idx))
        }
        TreeTarget::Vox(i) => {
            let Some(mut copy) = state.level.voxes.get(i).cloned() else {
                state.status = "tree: vox gone".into();
                return;
            };
            copy.off_x = off(copy.off_x, sx);
            state.level.voxes.push(copy);
            let idx = state.level.voxes.len() - 1;
            // A stamped vox writes real Map cells; rebuild so the copy renders.
            crate::vox::rebuild_map(state);
            state.status = format!("duplicated vox -> #{idx}");
            Some(TreeTarget::Vox(idx))
        }
        TreeTarget::ZoneInput(i) => {
            let Some(src) = state.level.zone_inputs.get(i).cloned() else {
                state.status = "tree: zone input gone".into();
                return;
            };
            let mut copy = src.clone();
            copy.name = format!("zone{}", state.level.zone_inputs.len());
            copy.x0 = off(src.x0, sx);
            copy.x1 = off(src.x1, sx);
            state.level.zone_inputs.push(copy);
            let idx = state.level.zone_inputs.len() - 1;
            state.status = format!("duplicated zone input -> #{idx}");
            Some(TreeTarget::ZoneInput(idx))
        }
        TreeTarget::Note(i) => {
            let Some(mut copy) = state.level.notes.get(i).cloned() else {
                state.status = "tree: note gone".into();
                return;
            };
            copy.x = off(copy.x, sx);
            state.level.notes.push(copy);
            let idx = state.level.notes.len() - 1;
            state.status = format!("duplicated note -> #{idx}");
            Some(TreeTarget::Note(idx))
        }
        TreeTarget::WallDecal(i) => {
            let Some(src) = state.level.decals.get(i).cloned() else {
                state.status = "tree: wall decal gone".into();
                return;
            };
            let mut copy = src.clone();
            // A unique name so the copy never collides on its `[wall_decal]` key,
            // and an x-offset so it never lands exactly on the source region.
            let mut k = state.level.decals.len() + 1;
            copy.name = loop {
                let candidate = format!("decal{k}");
                if !state.level.decals.iter().any(|d| d.name == candidate) {
                    break candidate;
                }
                k += 1;
            };
            copy.x0 = off(src.x0, sx);
            copy.x1 = off(src.x1, sx);
            state.level.decals.push(copy);
            let idx = state.level.decals.len() - 1;
            state.status = format!("duplicated wall decal -> #{idx}");
            Some(TreeTarget::WallDecal(idx))
        }
        TreeTarget::Holo(i) => {
            let Some(src) = state.level.holos.get(i).cloned() else {
                state.status = "tree: hologram gone".into();
                return;
            };
            let mut copy = src.clone();
            // A unique name so the copy never collides on its `[holo]` key, and an
            // x-offset so it never lands exactly on the source region.
            let mut k = state.level.holos.len() + 1;
            copy.name = loop {
                let candidate = format!("holo{k}");
                if !state.level.holos.iter().any(|h| h.name == candidate) {
                    break candidate;
                }
                k += 1;
            };
            copy.x0 = off(src.x0, sx);
            copy.x1 = off(src.x1, sx);
            state.level.holos.push(copy);
            let idx = state.level.holos.len() - 1;
            state.status = format!("duplicated hologram -> #{idx}");
            Some(TreeTarget::Holo(idx))
        }
        TreeTarget::Portal(i) => {
            let Some(src) = state.level.portals.get(i).cloned() else {
                state.status = "tree: portal gone".into();
                return;
            };
            let mut copy = src.clone();
            // A unique name so the copy never collides on its `[portal]` key, and an
            // x-offset on BOTH regions so neither lands exactly on the source.
            let mut k = state.level.portals.len() + 1;
            copy.name = loop {
                let candidate = format!("portal{k}");
                if !state.level.portals.iter().any(|p| p.name == candidate) {
                    break candidate;
                }
                k += 1;
            };
            copy.ax0 = off(src.ax0, sx);
            copy.ax1 = off(src.ax1, sx);
            copy.bx0 = off(src.bx0, sx);
            copy.bx1 = off(src.bx1, sx);
            state.level.portals.push(copy);
            let idx = state.level.portals.len() - 1;
            state.status = format!("duplicated portal -> #{idx}");
            Some(TreeTarget::Portal(idx))
        }
        TreeTarget::Zone(i) => {
            let Some(src) = state.level.overlay.zones.get(i).cloned() else {
                state.status = "tree: zone gone".into();
                return;
            };
            let mut copy = src.clone();
            let mut k = state.level.overlay.zones.len() + 1;
            copy.name = loop {
                let candidate = format!("zone{k}");
                if !state
                    .level
                    .overlay
                    .zones
                    .iter()
                    .any(|z| z.name == candidate)
                {
                    break candidate;
                }
                k += 1;
            };
            // x-offset the region so the copy never lands exactly on the source.
            if let re_core::Region::Box { min, max } = copy.region {
                copy.region = re_core::Region::Box {
                    min: (off(min.0, sx), min.1, min.2),
                    max: (off(max.0, sx), max.1, max.2),
                };
            }
            state.level.overlay.zones.push(copy);
            let idx = state.level.overlay.zones.len() - 1;
            state.status = format!("duplicated zone -> #{idx}");
            Some(TreeTarget::Zone(idx))
        }
        TreeTarget::Line(i) => {
            let Some(src) = state.level.overlay.lines.get(i).cloned() else {
                state.status = "tree: line gone".into();
                return;
            };
            let mut copy = src.clone();
            let mut k = state.level.overlay.lines.len() + 1;
            copy.name = loop {
                let candidate = format!("line{k}");
                if !state
                    .level
                    .overlay
                    .lines
                    .iter()
                    .any(|l| l.name == candidate)
                {
                    break candidate;
                }
                k += 1;
            };
            // x-offset every waypoint so the copy is visibly distinct.
            for p in &mut copy.path.points {
                p.x += 1.0;
            }
            state.level.overlay.lines.push(copy);
            let idx = state.level.overlay.lines.len() - 1;
            state.status = format!("duplicated line -> #{idx}");
            Some(TreeTarget::Line(idx))
        }
        TreeTarget::Inset(i) => {
            let Some(src) = state.level.overlay.insets.get(i).cloned() else {
                state.status = "tree: inset gone".into();
                return;
            };
            let mut copy = src.clone();
            let mut k = state.level.overlay.insets.len() + 1;
            copy.name = loop {
                let candidate = format!("inset{k}");
                if !state
                    .level
                    .overlay
                    .insets
                    .iter()
                    .any(|n| n.name == candidate)
                {
                    break candidate;
                }
                k += 1;
            };
            if let re_core::Region::Box { min, max } = copy.region {
                copy.region = re_core::Region::Box {
                    min: (off(min.0, sx), min.1, min.2),
                    max: (off(max.0, sx), max.1, max.2),
                };
            }
            state.level.overlay.insets.push(copy);
            let idx = state.level.overlay.insets.len() - 1;
            state.status = format!("duplicated inset -> #{idx}");
            Some(TreeTarget::Inset(idx))
        }
        TreeTarget::Part(i) => {
            let Some(mut copy) = state.level.parts.get(i).cloned() else {
                state.status = "tree: part gone".into();
                return;
            };
            // A unique name + an offset anchor so the copy never collides on its
            // key nor lands on top of the source. The mask is cloned verbatim.
            let mut k = state.level.parts.len() + 1;
            let name = loop {
                let candidate = format!("part{k}");
                if !state.level.parts.iter().any(|p| p.name == candidate) {
                    break candidate;
                }
                k += 1;
            };
            copy.name = name;
            copy.off_x = off(copy.off_x, sx);
            state.level.parts.push(copy);
            let idx = state.level.parts.len() - 1;
            state.status = format!("duplicated part -> #{idx}");
            Some(TreeTarget::Part(idx))
        }
        TreeTarget::Pipe(_) => {
            // A pipe is defined by its two endpoint nodes; a +1-offset copy would
            // duplicate the A->B wire (refused by `push_pipe_between`). Re-wire a new
            // pipe via the `P` flow instead.
            state.status = "pipe: re-wire with P (cannot offset-duplicate)".into();
            None
        }
        TreeTarget::Singleton(kind) => {
            state.status = format!("{} is one-per-level (cannot duplicate)", kind.label());
            None
        }
    };
    if let Some(t) = new_target {
        state.mark_dirty();
        select_target(state, t);
        // Re-point the tree cursor at the new copy's row so it stays highlighted.
        focus_target_row(state, t);
    }
}

/// Move the tree cursor onto the row whose object is `target` (after a duplicate
/// makes a new row). A no-match leaves the cursor where it is (then clamped).
fn focus_target_row(state: &mut EditorState, target: TreeTarget) {
    let rows = build_tree(state);
    if let Some(idx) = rows
        .iter()
        .position(|r| matches!(r, TreeRow::Object { target: t, .. } if *t == target))
    {
        state.tree.cursor = idx;
    }
    clamp_cursor(state);
}

// ───────────────────────────── type-selector palette ─────────────────────────────

/// The object TYPES the create palette offers, in menu order. Includes Vox
/// (vox-mode) alongside the [`EntityKind`]s so every placeable kind is one pick
/// away — the discoverable alternative to memorising the `o`-cycle order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PaletteItem {
    Kind(EntityKind),
    Vox,
}

/// The palette menu, in display order.
pub const PALETTE_ITEMS: [PaletteItem; 17] = [
    PaletteItem::Kind(EntityKind::Node),
    PaletteItem::Kind(EntityKind::Pipe),
    PaletteItem::Kind(EntityKind::Door),
    PaletteItem::Kind(EntityKind::Spawn),
    PaletteItem::Kind(EntityKind::Pedestal),
    PaletteItem::Kind(EntityKind::Part),
    PaletteItem::Kind(EntityKind::Emitter),
    PaletteItem::Kind(EntityKind::Trigger),
    PaletteItem::Kind(EntityKind::ZoneInput),
    PaletteItem::Kind(EntityKind::Note),
    PaletteItem::Kind(EntityKind::WallDecal),
    PaletteItem::Kind(EntityKind::Holo),
    PaletteItem::Kind(EntityKind::Portal),
    PaletteItem::Kind(EntityKind::Zone),
    PaletteItem::Kind(EntityKind::Line),
    PaletteItem::Kind(EntityKind::Inset),
    PaletteItem::Vox,
];

impl PaletteItem {
    /// Display label.
    pub fn label(self) -> &'static str {
        match self {
            PaletteItem::Kind(k) => k.label(),
            PaletteItem::Vox => "Vox",
        }
    }
}

/// Open (or close) the create palette — the object-TYPE SELECTOR. Mutually
/// exclusive with the focused tree panel. While open it captures Up/Down (move)
/// and Enter (pick → enter placement mode for that kind at the cursor).
pub fn toggle_palette(state: &mut EditorState) {
    if state.palette.is_some() {
        state.palette = None;
        state.status = "create palette closed".into();
    } else {
        state.tree.open = false;
        state.palette = Some(0);
        state.status = "create: Up/Dn pick type, Enter place, i close".into();
    }
}

/// Step the palette highlight by `delta` (±1), clamped.
pub fn palette_navigate(state: &mut EditorState, delta: i32) {
    if let Some(sel) = state.palette {
        let last = PALETTE_ITEMS.len() as i32 - 1;
        let next = (sel as i32 + delta).clamp(0, last);
        state.palette = Some(next as usize);
        state.status = format!("create: {}", PALETTE_ITEMS[next as usize].label());
    }
}

/// Pick the highlighted palette type: enter PLACEMENT mode for that kind (so the
/// next `space` creates one at the cursor), and close the palette. This replaces
/// having to remember the `o`-cycle order.
pub fn palette_pick(state: &mut EditorState) {
    let Some(sel) = state.palette else { return };
    let item = PALETTE_ITEMS[sel.min(PALETTE_ITEMS.len() - 1)];
    state.palette = None;
    match item {
        PaletteItem::Kind(kind) => {
            state.vox_mode = false;
            state.material_mode = false;
            state.place = Some(kind);
            // Reset focus to the kind's first field (no stale cross-kind focus).
            if let Some(f) = crate::state::entity_field_default(kind) {
                state.entity_field = f;
            }
            state.status = if kind == EntityKind::Pipe {
                "place: Pipe — cursor on SOURCE node, P to start, then TARGET".into()
            } else {
                format!("place: {} (space to create)", kind.label())
            };
        }
        PaletteItem::Vox => {
            state.place = None;
            state.material_mode = false;
            state.vox_mode = true;
            state.status = "place: Vox (space to stamp)".into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::EmitterOrigin;

    /// A small populated level: 2 nodes, a door, a spawn, a pedestal, an emitter,
    /// a trigger, and a vox stamp — one of every group.
    fn populated() -> EditorState {
        let mut s = EditorState::new_empty(16, 16, 4);
        let a = s.level.push_node(1, 5, 2);
        s.level.nodes[a].is_input = true;
        let b = s.level.push_node(10, 3, 1);
        s.level.nodes[b].is_output = true;
        s.level.door.x = 11;
        s.level.door.y = 3;
        s.level.door.z = 1;
        s.level.player.x = 2;
        s.level.player.y = 2;
        s.level.player.z = 1;
        s.level.pedestal.cx = 3;
        s.level.pedestal.cy = 3;
        s.level.pedestal.w = 1;
        s.level.pedestal.h = 1;
        s.level.pedestal.d = 1;
        let mut e = re_world::EmitterSpec::default()
            .with_origin(EmitterOrigin::World(glam::Vec3::new(4.5, 4.5, 1.5)));
        e.validate();
        s.level.emitters.push(e);
        s.level.triggers.push(re_core::Trigger {
            region: re_core::Region::Box {
                min: (6, 6, 0),
                max: (7, 7, 0),
            },
            action: re_core::TriggerAction::event("event"),
            once: true,
        });
        s.level.voxes.push(re_level::LevelVox {
            name: "v0".into(),
            file: "levels/x.vox".into(),
            off_x: 8,
            off_y: 8,
            off_z: 0,
            material: re_level::VoxMaterial::Solid,
            scale: 1,
        });
        s
    }

    #[test]
    fn tree_enumerates_all_groups_with_correct_counts() {
        let s = populated();
        assert_eq!(group_count(&s, TreeGroup::Nodes), 2);
        assert_eq!(group_count(&s, TreeGroup::Doors), 1);
        assert_eq!(group_count(&s, TreeGroup::Spawn), 1);
        assert_eq!(group_count(&s, TreeGroup::Pedestal), 1);
        assert_eq!(group_count(&s, TreeGroup::Emitters), 1);
        assert_eq!(group_count(&s, TreeGroup::Triggers), 1);
        assert_eq!(group_count(&s, TreeGroup::Vox), 1);
        assert_eq!(object_count(&s), 8);
    }

    #[test]
    fn empty_level_shows_only_group_headers_and_summary() {
        let s = EditorState::new_empty(8, 8, 4);
        let rows = build_tree(&s);
        // Summary + one header per group, no object rows.
        assert_eq!(rows.len(), 1 + TREE_GROUPS.len());
        assert!(matches!(rows[0], TreeRow::Summary { total: 0 }));
        assert!(rows[1..].iter().all(|r| matches!(r, TreeRow::Group { .. })));
    }

    #[test]
    fn tree_rows_carry_object_coords() {
        let s = populated();
        let rows = build_tree(&s);
        // The first node row should mention its coords.
        let node_row = rows.iter().find_map(|r| match r {
            TreeRow::Object {
                target: TreeTarget::Node(0),
                text,
                ..
            } => Some(text.clone()),
            _ => None,
        });
        let text = node_row.expect("node 0 row present");
        assert!(text.contains("(1,5,2)"), "node row shows coords: {text}");
        assert!(text.contains("input"), "node row shows io role: {text}");
    }

    #[test]
    fn navigation_steps_and_clamps() {
        let mut s = populated();
        s.tree.cursor = 0;
        navigate(&mut s, -1); // clamp at top
        assert_eq!(s.tree.cursor, 0);
        navigate(&mut s, 1);
        assert_eq!(s.tree.cursor, 1);
        // Walk to the very end; cursor clamps at the last row.
        let last = build_tree(&s).len() - 1;
        for _ in 0..100 {
            navigate(&mut s, 1);
        }
        assert_eq!(s.tree.cursor, last);
    }

    #[test]
    fn collapse_hides_children() {
        let mut s = populated();
        let before = build_tree(&s).len();
        // Fold Nodes (2 children) → list shrinks by 2.
        s.tree.collapsed[TreeGroup::Nodes.index()] = true;
        let after = build_tree(&s).len();
        assert_eq!(after, before - 2, "folding nodes hides its 2 children");
    }

    #[test]
    fn collapse_from_object_row_folds_enclosing_group() {
        let mut s = populated();
        // Put the cursor on the first NODE object row, then fold.
        let rows = build_tree(&s);
        let node_row = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Node(0),
                        ..
                    }
                )
            })
            .unwrap();
        s.tree.cursor = node_row;
        toggle_collapse(&mut s);
        assert!(
            s.tree.collapsed[TreeGroup::Nodes.index()],
            "node group folded"
        );
    }

    #[test]
    fn select_node_sets_selector_cursor_and_camera() {
        let mut s = populated();
        select_target(&mut s, TreeTarget::Node(0));
        assert_eq!(s.place, Some(EntityKind::Node));
        assert_eq!(s.selected_node, Some(0));
        assert_eq!(s.cursor, (1, 5));
        assert_eq!(s.z, 2);
        // Camera jumped to the cell centre.
        assert_eq!(s.cam.pos.x, 1.5);
        assert_eq!(s.cam.pos.y, 5.5);
    }

    #[test]
    fn select_vox_enters_vox_mode() {
        let mut s = populated();
        select_target(&mut s, TreeTarget::Vox(0));
        assert!(s.vox_mode);
        assert_eq!(s.selected_vox, Some(0));
        assert!(s.place.is_none());
        assert_eq!(s.cursor, (8, 8));
    }

    #[test]
    fn select_door_singleton_focuses_and_jumps() {
        let mut s = populated();
        select_target(&mut s, TreeTarget::Singleton(EntityKind::Door));
        assert_eq!(s.place, Some(EntityKind::Door));
        assert_eq!(s.cursor, (11, 3));
        assert_eq!(s.z, 1);
    }

    #[test]
    fn duplicate_node_makes_distinct_copy_and_selects_it() {
        let mut s = populated();
        let before = s.level.nodes.len();
        // Cursor on node 0's row.
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Node(0),
                        ..
                    }
                )
            })
            .unwrap();
        duplicate_current(&mut s);
        assert_eq!(s.level.nodes.len(), before + 1, "a node was added");
        let new = s.level.nodes.len() - 1;
        // The copy carries the source's io role but a UNIQUE name + offset pos.
        assert!(s.level.nodes[new].is_input);
        assert_ne!(s.level.nodes[new].name, s.level.nodes[0].name);
        assert_eq!((s.level.nodes[new].x, s.level.nodes[new].y), (2, 6));
        // The copy is selected.
        assert_eq!(s.selected_node, Some(new));
    }

    #[test]
    fn duplicate_node_at_far_edge_lands_on_a_different_cell() {
        // A 16x16x4 world: the last valid cell on each axis is index 15 (x/y)
        // and 3 (z). Place a node in the far corner so +1 would collapse onto it.
        let mut s = EditorState::new_empty(16, 16, 4);
        let src = s.level.push_node(15, 15, 3);
        let (sx, sy, sz) = s.map.size();
        s.selected_node = Some(src);
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Node(0),
                        ..
                    }
                )
            })
            .unwrap();
        duplicate_current(&mut s);
        let new = s.level.nodes.len() - 1;
        let (nx, ny, nz) = (
            s.level.nodes[new].x,
            s.level.nodes[new].y,
            s.level.nodes[new].z,
        );
        // The copy must NOT sit on the source cell …
        assert_ne!(
            (nx, ny, nz),
            (15, 15, 3),
            "copy must not collapse onto source"
        );
        // … and must stay in bounds (here: stepped back to 14,14,2).
        assert!(nx >= 0 && (nx as usize) < sx, "x in bounds: {nx}");
        assert!(ny >= 0 && (ny as usize) < sy, "y in bounds: {ny}");
        assert!(nz >= 0 && (nz as usize) < sz, "z in bounds: {nz}");
        assert_eq!((nx, ny, nz), (14, 14, 2), "far-edge copy steps back by 1");
    }

    #[test]
    fn duplicate_node_with_room_ahead_offsets_by_plus_one() {
        let mut s = EditorState::new_empty(16, 16, 4);
        let src = s.level.push_node(5, 6, 1);
        s.selected_node = Some(src);
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Node(0),
                        ..
                    }
                )
            })
            .unwrap();
        duplicate_current(&mut s);
        let new = s.level.nodes.len() - 1;
        let n = &s.level.nodes[new];
        assert_eq!(
            (n.x, n.y, n.z),
            (6, 7, 2),
            "with room ahead, offset is +1 per axis"
        );
    }

    #[test]
    fn duplicate_emitter_offsets_origin() {
        let mut s = populated();
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Emitter(0),
                        ..
                    }
                )
            })
            .unwrap();
        let before = s.level.emitters.len();
        duplicate_current(&mut s);
        assert_eq!(s.level.emitters.len(), before + 1);
        // The copy's origin x is nudged by +1.
        if let (EmitterOrigin::World(a), EmitterOrigin::World(b)) = (
            &s.level.emitters[0].origin,
            &s.level.emitters[before].origin,
        ) {
            assert!((b.x - a.x - 1.0).abs() < 1e-3, "origin x offset by +1");
        } else {
            panic!("expected world origins");
        }
    }

    #[test]
    fn duplicate_singleton_is_refused() {
        let mut s = populated();
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Singleton(EntityKind::Door),
                        ..
                    }
                )
            })
            .unwrap();
        let before = s.level.nodes.len();
        duplicate_current(&mut s);
        // No new node/anything; status explains it.
        assert_eq!(s.level.nodes.len(), before);
        assert!(s.status.contains("one-per-level"));
    }

    #[test]
    fn delete_from_tree_removes_object() {
        let mut s = populated();
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Node(1),
                        ..
                    }
                )
            })
            .unwrap();
        let before = s.level.nodes.len();
        delete_current(&mut s);
        assert_eq!(s.level.nodes.len(), before - 1, "node removed via tree");
    }

    #[test]
    fn delete_emitter_from_tree() {
        let mut s = populated();
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Emitter(0),
                        ..
                    }
                )
            })
            .unwrap();
        delete_current(&mut s);
        assert_eq!(s.level.emitters.len(), 0, "emitter removed via tree");
    }

    #[test]
    fn palette_pick_enters_placement_mode() {
        let mut s = EditorState::new_empty(8, 8, 4);
        toggle_palette(&mut s);
        assert_eq!(s.palette, Some(0));
        // Default highlight is Node; pick it.
        palette_pick(&mut s);
        assert_eq!(s.place, Some(EntityKind::Node));
        assert!(s.palette.is_none(), "palette closes after pick");
    }

    #[test]
    fn palette_navigate_to_vox_and_pick() {
        let mut s = EditorState::new_empty(8, 8, 4);
        toggle_palette(&mut s);
        // Step to the last item (Vox).
        for _ in 0..PALETTE_ITEMS.len() {
            palette_navigate(&mut s, 1);
        }
        assert_eq!(s.palette, Some(PALETTE_ITEMS.len() - 1));
        palette_pick(&mut s);
        assert!(s.vox_mode, "picking Vox enters vox mode");
        assert!(s.place.is_none());
    }

    #[test]
    fn nested_children_render_with_increasing_depth() {
        use re_level::{HierKind, HierRef};
        let mut s = populated();
        // Add a second + third node so we have node 0/1/2 to chain.
        let n2 = s.level.push_node(12, 4, 1);
        // node1 under node0, node2 under node1 (a 3-deep chain within Nodes group).
        let a = HierRef::new(HierKind::Node, 0);
        let b = HierRef::new(HierKind::Node, 1);
        let c = HierRef::new(HierKind::Node, n2);
        s.level.set_hier_parent(b, a);
        s.level.set_hier_parent(c, b);
        let rows = build_tree(&s);
        let depth_of = |target: TreeTarget| {
            rows.iter().find_map(|r| match r {
                TreeRow::Object {
                    target: t, depth, ..
                } if *t == target => Some(*depth),
                _ => None,
            })
        };
        assert_eq!(depth_of(TreeTarget::Node(0)), Some(0), "root at depth 0");
        assert_eq!(depth_of(TreeTarget::Node(1)), Some(1), "child at depth 1");
        assert_eq!(
            depth_of(TreeTarget::Node(n2)),
            Some(2),
            "grandchild at depth 2"
        );
        // And the child rows are ordered AFTER the parent within the group.
        let pos = |target: TreeTarget| {
            rows.iter()
                .position(|r| matches!(r, TreeRow::Object { target: t, .. } if *t == target))
                .unwrap()
        };
        assert!(pos(TreeTarget::Node(0)) < pos(TreeTarget::Node(1)));
        assert!(pos(TreeTarget::Node(1)) < pos(TreeTarget::Node(n2)));
    }

    #[test]
    fn flat_tree_keeps_all_depths_zero() {
        // With no hierarchy, every object is a root (depth 0) — back-compat.
        let s = populated();
        let rows = build_tree(&s);
        assert!(rows
            .iter()
            .all(|r| !matches!(r, TreeRow::Object { depth, .. } if *depth != 0)));
    }

    #[test]
    fn tree_enumerates_pipes_with_endpoints_and_state() {
        let mut s = populated();
        // Wire node0 -> node1 (the two nodes `populated` already places).
        s.level.push_pipe_between(0, 1).expect("pipe wired");
        assert_eq!(group_count(&s, TreeGroup::Pipes), 1);
        let rows = build_tree(&s);
        let pipe_row = rows.iter().find_map(|r| match r {
            TreeRow::Object {
                target: TreeTarget::Pipe(0),
                text,
                ..
            } => Some(text.clone()),
            _ => None,
        });
        let text = pipe_row.expect("pipe 0 row present");
        // The row shows A->B and the off state.
        let a = &s.level.nodes[0].name;
        let b = &s.level.nodes[1].name;
        assert!(
            text.contains(&format!("{a}->{b}")),
            "pipe row shows endpoints: {text}"
        );
        assert!(text.contains("off"), "pipe row shows state: {text}");
    }

    #[test]
    fn select_pipe_from_tree_focuses_and_jumps_to_source() {
        let mut s = populated();
        s.level.push_pipe_between(0, 1).expect("pipe wired");
        select_target(&mut s, TreeTarget::Pipe(0));
        assert_eq!(s.place, Some(EntityKind::Pipe));
        assert_eq!(s.selected_pipe, Some(0));
        // Jumped to the source node cell (node 0 @ (1,5,2)).
        assert_eq!(s.cursor, (1, 5));
        assert_eq!(s.z, 2);
    }

    #[test]
    fn delete_pipe_from_tree_removes_it() {
        let mut s = populated();
        s.level.push_pipe_between(0, 1).expect("pipe wired");
        let rows = build_tree(&s);
        s.tree.cursor = rows
            .iter()
            .position(|r| {
                matches!(
                    r,
                    TreeRow::Object {
                        target: TreeTarget::Pipe(0),
                        ..
                    }
                )
            })
            .unwrap();
        delete_current(&mut s);
        assert_eq!(s.level.pipes.len(), 0, "pipe removed via tree");
    }

    #[test]
    fn toggle_tree_and_palette_are_mutually_exclusive() {
        let mut s = EditorState::new_empty(8, 8, 4);
        toggle(&mut s); // open tree
        assert!(s.tree.open);
        toggle_palette(&mut s); // opening palette closes tree
        assert!(!s.tree.open);
        assert!(s.palette.is_some());
        toggle(&mut s); // opening tree closes palette
        assert!(s.tree.open);
        assert!(s.palette.is_none());
    }
}
