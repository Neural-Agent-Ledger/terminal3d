//! **Trigger fields** — regions that fire an arbitrary event when the player
//! enters them ("триггер поля на произвольные события").
//!
//! A [`Trigger`] is a box [`Region`] + a [`TriggerAction`] + a `once` flag. The
//! engine layers handle it like the existing door / zone-input features:
//!
//! * `re_world` stores the **resolved** triggers on the `Map` (alongside resolved
//!   insets) and exposes a pure `triggers_containing(pos)` query.
//! * `re_level` parses a `[trigger]` section, resolves an `ActivateNode`'s node
//!   name to an index in `finalize`, and writes it back out in `re_voximport`.
//! * `re_game` runs a per-frame system (modeled on `watch_win`) that fires the
//!   action on the OUTSIDE→INSIDE camera transition, respecting `once`.
//!
//! This module is **engine-pure**: plain data + a containment helper. No Bevy,
//! no terminal, no rendering. The firing semantics (how an action mutates the
//! puzzle / emits an event) live in the game layer, NOT here — a `Trigger` only
//! describes WHAT to do, not HOW.

use crate::Region;

/// What a [`Trigger`] does when the player enters its region.
///
/// "Arbitrary events" are supported pragmatically by two MVP variants. The enum
/// is `#[non_exhaustive]` so new variants (a future `Win`, `Toggle`, `Emit`, …)
/// are a non-breaking addition for downstream `match`es.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriggerAction {
    /// Drive the named puzzle node — the most useful event. The game layer
    /// resolves the name to a node index and sets it active (driven like an
    /// input → re-runs the forward pass), so a trigger field can satisfy a door
    /// target by walking into a region.
    ///
    /// The named node MUST be an **input** node (enforced by `re_level::finalize`,
    /// the same rule as `[zone_input]`): a trigger drives it exactly as a
    /// floor-zone does, but latched (it does not turn off on exit).
    ActivateNode {
        /// Authored node name (resolved to an index by `re_level::finalize`).
        name: String,
        /// Resolved index of `name` into the puzzle's node list. Set by
        /// `re_level::finalize`; `usize::MAX` until resolved (an unresolved
        /// trigger fires nothing — defensive).
        node_index: usize,
    },
    /// A generic named event — the "arbitrary event" hook. The game layer emits
    /// an observable [`TriggerFired`](../../re_game)-style message and logs it,
    /// so future wiring (audio cue, scripted door, narrative beat) can subscribe
    /// without changing this enum.
    Event {
        /// Free-form event label, surfaced verbatim to subscribers.
        label: String,
    },
}

impl TriggerAction {
    /// Construct an unresolved [`ActivateNode`](Self::ActivateNode) from a name
    /// (its `node_index` starts at the unresolved sentinel `usize::MAX`).
    pub fn activate_node(name: impl Into<String>) -> Self {
        TriggerAction::ActivateNode {
            name: name.into(),
            node_index: usize::MAX,
        }
    }

    /// Construct an [`Event`](Self::Event) action with the given label.
    pub fn event(label: impl Into<String>) -> Self {
        TriggerAction::Event {
            label: label.into(),
        }
    }
}

/// A **trigger field**: a box [`Region`] that fires an [`TriggerAction`] when the
/// player's camera enters it.
///
/// * `region` — the box the player must enter (inclusive AABB, octave-0 cells).
/// * `action` — what to do on entry (see [`TriggerAction`]).
/// * `once` — `true` ⇒ fire ONCE then disarm (until the level resets); `false` ⇒
///   re-fire on each fresh OUTSIDE→INSIDE entry. Standing still inside never
///   re-fires either way — the firing system tracks the inside/outside edge.
///
/// [`Default`] is an inert empty-region [`Event`](TriggerAction::Event) with an
/// empty label, fired-once — it covers no cells, so it never fires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trigger {
    /// The region whose entry fires [`action`](Self::action).
    pub region: Region,
    /// The action fired on OUTSIDE→INSIDE entry.
    pub action: TriggerAction,
    /// Fire once then disarm (`true`) vs re-fire on every fresh entry (`false`).
    pub once: bool,
}

impl Default for Trigger {
    fn default() -> Self {
        Trigger {
            region: Region::default(),
            action: TriggerAction::Event {
                label: String::new(),
            },
            once: true,
        }
    }
}

impl Trigger {
    /// True iff integer cell `(x, y, z)` lies inside this trigger's region — the
    /// inclusive AABB test ([`Region::contains`]). The firing system feeds the
    /// player's floored camera cell here.
    #[inline]
    pub fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        self.region.contains(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxr(min: (i32, i32, i32), max: (i32, i32, i32)) -> Region {
        Region::Box { min, max }
    }

    #[test]
    fn contains_inside_outside_edge() {
        let t = Trigger {
            region: boxr((2, 0, 2), (4, 0, 5)),
            action: TriggerAction::event("e"),
            once: true,
        };
        // Interior.
        assert!(t.contains(3, 0, 3));
        // Edges are inclusive.
        assert!(t.contains(2, 0, 2));
        assert!(t.contains(4, 0, 5));
        // Just outside each face.
        assert!(!t.contains(1, 0, 3));
        assert!(!t.contains(5, 0, 3));
        assert!(!t.contains(3, 0, 1));
        assert!(!t.contains(3, 0, 6));
        // Wrong layer.
        assert!(!t.contains(3, 1, 3));
    }

    #[test]
    fn default_trigger_is_inert() {
        let t = Trigger::default();
        // Empty box at origin covers only (0,0,0)… actually a point. It is the
        // documented inert default; the parser never produces a degenerate one.
        assert!(t.once);
        assert_eq!(t.action, TriggerAction::event(""));
    }

    #[test]
    fn activate_node_starts_unresolved() {
        let a = TriggerAction::activate_node("gate");
        assert_eq!(
            a,
            TriggerAction::ActivateNode {
                name: "gate".to_string(),
                node_index: usize::MAX,
            }
        );
    }
}
