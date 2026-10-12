//! `state_bus` — the cast-scoped binding of the [`CellStateMap`]
//! into the deep shade path.
//!
//! The state bus lets a material react to the puzzle's
//! live state. The per-cell lookup happens in [`shade_cell`](crate::shade_cell) —
//! the single place that knows a cell's `material` tag — but the cast entry point
//! (`composite_render_sys`) is many frames up the call stack, and threading an
//! `Option<&CellStateMap>` through every `shade_cell` call + cast helper would be a
//! large multi-file API churn. So, EXACTLY like the render-config knobs in
//! [`crate::config`], the active map is published for the current cast via a
//! thread-local that the shade path reads back.
//!
//! ## Purity + safety
//!
//! This module reads ONLY [`re_core::CellStateMap`] (generic `u32`/`f32` per tag) —
//! no `re_puzzle`, no puzzle semantics — so `re_render` keeps zero dependency on
//! `re_puzzle`. And it holds an OWNED clone of the map in the thread-local (no raw
//! pointers, no `unsafe`): the state map is tiny (a handful of puzzle-node tags), so
//! cloning it once per cast is negligible, and the borrow checker guarantees the
//! stored value outlives every read.
//!
//! Default (no map installed) ⇒ an EMPTY map: `state(tag) == 0`, `pulse(tag) == 0`
//! for every tag → today's frame, byte-for-byte.

use std::cell::RefCell;

use re_core::CellStateMap;

thread_local! {
    /// The [`CellStateMap`] active for the current cast on this thread. Empty by
    /// default (⇒ every tag reads `state 0` / `pulse 0`, today's look). Installed by
    /// [`set_active`] / cleared by [`clear_active`] at the cast boundary, or scoped by
    /// [`CellStateScope`] in tests.
    static ACTIVE: RefCell<CellStateMap> = const { RefCell::new(CellStateMap::new()) };
}

/// Install `map` (by clone) as the active state bus for this thread. Called by the
/// cast entry once per frame from the live [`CellStateMap`] resource. The clone is
/// cheap (a few node tags). Pair with [`clear_active`] (or use [`CellStateScope`]).
#[inline]
pub fn set_active(map: &CellStateMap) {
    ACTIVE.with(|c| c.borrow_mut().clone_from(map));
}

/// Reset the active state bus to EMPTY (today's look) — call after a cast so a later
/// cast without a map on this thread is unaffected by a stale one.
#[inline]
pub fn clear_active() {
    ACTIVE.with(|c| c.borrow_mut().clear());
}

/// Look up the state word for `tag` from the cast's active state bus, or `0` when the
/// map is empty (the default → today's look). Pure, allocation-free.
#[inline]
pub fn state_of(tag: u16) -> u32 {
    ACTIVE.with(|c| c.borrow().state(tag))
}

/// Look up the transient pulse for `tag` from the cast's active state bus, or `0.0`
/// when the map is empty. Pure, allocation-free.
#[inline]
pub fn pulse_of(tag: u16) -> f32 {
    ACTIVE.with(|c| c.borrow().pulse(tag))
}

/// An RAII test/scoped helper: install `map` as the active bus, restoring the
/// previous map on drop. Handy in unit tests (and any nested cast) so the thread is
/// left clean. Production casts use [`set_active`] + [`clear_active`] directly.
pub struct CellStateScope {
    prev: CellStateMap,
}

impl CellStateScope {
    /// Install `map` (by clone) as the active bus until the guard drops.
    #[inline]
    pub fn new(map: &CellStateMap) -> Self {
        let prev = ACTIVE.with(|c| c.borrow().clone());
        set_active(map);
        CellStateScope { prev }
    }
}

impl Drop for CellStateScope {
    #[inline]
    fn drop(&mut self) {
        // Restore the previous map (runs on unwind too).
        ACTIVE.with(|c| c.borrow_mut().clone_from(&self.prev));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_map_reads_empty_default() {
        clear_active();
        assert_eq!(state_of(0), 0);
        assert_eq!(state_of(42), 0);
        assert_eq!(pulse_of(7), 0.0);
    }

    #[test]
    fn scope_publishes_the_map_then_clears_on_drop() {
        clear_active();
        let mut m = CellStateMap::new();
        m.set_state(3, 0xBEEF);
        m.set_pulse(3, 0.5);
        {
            let _scope = CellStateScope::new(&m);
            assert_eq!(state_of(3), 0xBEEF);
            assert_eq!(pulse_of(3), 0.5);
            assert_eq!(state_of(2), 0, "untouched tag still defaults");
        }
        // Dropped ⇒ back to the empty default.
        assert_eq!(state_of(3), 0);
        assert_eq!(pulse_of(3), 0.0);
    }

    #[test]
    fn nested_scopes_restore_the_outer_map() {
        clear_active();
        let mut outer = CellStateMap::new();
        outer.set_state(1, 10);
        let mut inner = CellStateMap::new();
        inner.set_state(1, 20);
        let _o = CellStateScope::new(&outer);
        assert_eq!(state_of(1), 10);
        {
            let _i = CellStateScope::new(&inner);
            assert_eq!(state_of(1), 20);
        }
        assert_eq!(state_of(1), 10, "inner drop restores the outer map");
    }
}
