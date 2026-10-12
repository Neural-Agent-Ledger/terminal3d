//! Bounded snapshot-based UNDO / REDO for the editor.
//!
//! Design — **snapshot, not granular op**. Before any mutating edit we clone the
//! editable state `(Map, Level)` and push it on the undo stack; a new edit clears
//! the redo stack. `undo` pops the undo stack, swaps the current state for the
//! popped one, and pushes the *previous* current onto redo; `redo` is the mirror.
//! This is O(map+level) per edit, which is trivial at editor map scale (tens of
//! thousands of cells) and is the simplest provably-correct scheme — every edit
//! path is covered as long as it calls `EditorState::push_undo` first.
//!
//! Future optimisation (NOT done here): record granular per-cell/per-entity ops so
//! a snapshot isn't cloned on every stroke. Deferred — snapshot is correct first.
//!
//! Keys (documented here + in the on-screen help):
//! * `u`      → UNDO  (pop undo → apply, push current onto redo)
//! * `y`      → REDO  (pop redo → apply, push current onto undo)
//! * Snapshots are taken at the single mutation choke points (`apply_tool_at_cursor`
//!   and the discrete mutating branches in `edit::apply_key`), never on navigation.
use re_level::Level;
use re_world::Map;

/// One captured editor state: the geometry [`Map`] and the entity/material/etc.
/// [`Level`]. The cursor / camera / view-mode are deliberately NOT captured — undo
/// restores *content*, not where you were looking.
#[derive(Clone)]
pub struct Snapshot {
    pub map: Map,
    pub level: Level,
}

impl Snapshot {
    /// The canonical `.lvl` text for this snapshot's `(map, level)`. `Map` and
    /// `Level` do not implement `PartialEq` (they live in sibling crates we do not
    /// own), so the COMMIT-IF-CHANGED path compares snapshots by their serialized
    /// document — the exact "byte-identical" measure the save path uses. Cheap at
    /// human-speed keystroke rate; clones so it never mutates the live state.
    pub fn canonical(&self) -> String {
        let mut level = self.level.clone();
        level.cells = Some(re_level::CellGrid::from_map(&self.map));
        re_voximport::to_lvl_text(&level)
    }

    /// True when this snapshot serializes to the SAME canonical document as `other`
    /// — i.e. the `(map, level)` content is byte-identical.
    pub fn same_document(&self, other: &Snapshot) -> bool {
        self.canonical() == other.canonical()
    }
}

/// Maximum number of undo (and redo) snapshots retained. Oldest is dropped when
/// the cap is exceeded — bounded memory regardless of session length.
pub const HISTORY_CAP: usize = 64;

/// Bounded undo + redo stacks. Lives on [`crate::state::EditorState`].
#[derive(Default)]
pub struct History {
    /// States we can return TO (most-recent on top).
    undo: Vec<Snapshot>,
    /// States we can re-apply after undoing (most-recent on top). Cleared on any
    /// fresh edit.
    redo: Vec<Snapshot>,
}

impl History {
    /// Record `snap` as the pre-edit state and invalidate the redo branch. Caps
    /// the undo stack at [`HISTORY_CAP`] by dropping the oldest entry.
    pub fn push(&mut self, snap: Snapshot) {
        self.undo.push(snap);
        if self.undo.len() > HISTORY_CAP {
            // Drop the oldest (front). `remove(0)` is fine for a 64-cap vec.
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Pop the most-recent undo snapshot, if any, given the `current` state to be
    /// pushed onto the redo stack. Returns the snapshot to apply.
    pub fn undo(&mut self, current: Snapshot) -> Option<Snapshot> {
        let prev = self.undo.pop()?;
        self.redo.push(current);
        if self.redo.len() > HISTORY_CAP {
            self.redo.remove(0);
        }
        Some(prev)
    }

    /// Pop the most-recent redo snapshot, if any, given the `current` state to be
    /// pushed back onto the undo stack. Returns the snapshot to apply.
    pub fn redo(&mut self, current: Snapshot) -> Option<Snapshot> {
        let next = self.redo.pop()?;
        self.undo.push(current);
        if self.undo.len() > HISTORY_CAP {
            self.undo.remove(0);
        }
        Some(next)
    }

    /// Like [`push`](Self::push) but does NOT clear the redo branch. The
    /// COMMIT-IF-CHANGED checkpoint path uses this so an operation that turns out
    /// to be a NO-OP does not destroy the redo stack: redo is invalidated only once
    /// the edit is confirmed to have changed the document, via [`clear_redo`](Self::clear_redo). Still
    /// caps the undo stack by dropping the oldest entry.
    pub fn push_checkpoint(&mut self, snap: Snapshot) {
        self.undo.push(snap);
        if self.undo.len() > HISTORY_CAP {
            self.undo.remove(0);
        }
    }

    /// Invalidate the redo branch. Called when a COMMIT-IF-CHANGED checkpoint is
    /// KEPT (i.e. the operation really changed the document) — a fresh edit makes
    /// the previously-undone states unreachable, matching [`push`](Self::push)'s semantics.
    pub fn clear_redo(&mut self) {
        self.redo.clear();
    }

    /// A borrow of the most-recent undo snapshot (the one [`push`](Self::push) just placed on
    /// top), if any. Used by the COMMIT-IF-CHANGED path to compare the pre-edit
    /// snapshot against the post-handler state without popping it.
    pub fn top_undo(&self) -> Option<&Snapshot> {
        self.undo.last()
    }

    /// Discard the most-recent undo snapshot (the top of the undo stack) WITHOUT
    /// touching the redo stack. Used to roll back a checkpoint pushed for an
    /// operation that turned out to be a no-op, so a dead snapshot never stacks up.
    /// No-op when the undo stack is empty.
    pub fn pop_undo(&mut self) {
        self.undo.pop();
    }

    /// How many undo steps are available.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// How many redo steps are available.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Drop all history (used when a fresh level replaces the document, e.g. the
    /// new-level wizard) so stale snapshots can't restore the discarded map.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}
