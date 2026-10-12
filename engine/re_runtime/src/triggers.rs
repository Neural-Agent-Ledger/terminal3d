//! `TriggerFired`, `TriggerRuntime`, `TriggerStates`, `FiredTrigger`,
//! `apply_triggers`, `apply_zone_inputs`. Pure over [`re_core::Trigger`] /
//! [`re_level::ZoneInput`] / [`re_puzzle::PuzzleState`] /
//! [`re_world::CameraPose`]; no seams.
//!
//! # What stays in the host
//!
//! Only the pure firing core and its plain-data types live here. The Bevy
//! systems that WIRE this into a running app — the per-frame `trigger_sys` /
//! `zone_input_sys` that read the live [`re_world::CameraPose`] and
//! [`re_world::Map`], plus a subscriber step that turns a fired label into an
//! audio stinger and a narrative-beat string — stay in the host, because that
//! step is genuinely game content (which sound plays, which line of story
//! text shows). A host wires this module's [`apply_triggers`] /
//! [`apply_zone_inputs`] into its own systems: rebuild [`TriggerStates`] on
//! every level (re)load, call the pure function each tick, and surface the
//! returned [`FiredTrigger`]s however it likes (a message, a log line, both).

use bevy::prelude::Resource;
use re_core::{Trigger, TriggerAction};
use re_level::ZoneInput;
use re_puzzle::{set_input_value, PuzzleState};
use re_world::CameraPose;

/// Emitted when a [`re_core::Trigger`] field FIRES — the generic "arbitrary
/// event" surface for the trigger-fields feature. `label` is:
///
/// * the `Event` action's authored label, for a `TriggerAction::Event`, or
/// * the activated node's NAME, for a `TriggerAction::ActivateNode` (so a
///   subscriber sees which node a walk-in lit, even though the puzzle side-effect
///   already happened).
///
/// A host that wants to react to a fire (audio cue, scripted door, narrative
/// beat) subscribes to this message instead of growing the action enum.
/// Engine-agnostic: a plain Bevy message, observable in headless tests.
#[derive(bevy::ecs::message::Message, Clone, Debug, PartialEq, Eq)]
pub struct TriggerFired {
    /// The fired event's label (Event label, or activated node name).
    pub label: String,
}

/// Per-trigger runtime bookkeeping (parallel to the map's resolved triggers).
/// Tracks the inside/outside edge so a trigger fires on ENTRY only (never every
/// frame while the player stands inside), plus the `once` disarm latch.
#[derive(Clone, Copy, Debug, Default)]
pub struct TriggerRuntime {
    /// Was the player inside this trigger's region on the PREVIOUS evaluation?
    /// The fire edge is `!inside_last && inside_now`.
    inside_last: bool,
    /// Has this `once` trigger already fired? A `once` trigger with `fired`
    /// set is disarmed until the level reset re-arms it.
    fired: bool,
}

/// Live per-level trigger-field state. Rebuilt (one [`TriggerRuntime`] per
/// resolved trigger, all dis-armed/outside) on every level (re-)load by the
/// host's own level-enter handler, so a reset re-arms every trigger exactly
/// like a win watcher re-arms the win. Empty ⇒ the level has no `[trigger]`
/// section (the firing system early-returns).
#[derive(Resource, Default)]
pub struct TriggerStates(pub Vec<TriggerRuntime>);

/// One action a [`Trigger`] decided to FIRE this frame, returned by
/// [`apply_triggers`] for the caller to surface (emit a message, log). The
/// puzzle side-effect (an `ActivateNode`) is ALREADY applied by
/// [`apply_triggers`]; this only carries the observable label out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiredTrigger {
    /// The label to surface on a [`TriggerFired`] message (see its docs).
    pub label: String,
}

/// PURE firing core (headless-testable): for every resolved trigger, detect the
/// OUTSIDE→INSIDE entry edge against the camera's floored cell, fire the action,
/// and update the runtime state. Returns the list of fired labels (in trigger
/// order) so the caller can emit one message per fire.
///
/// Rules:
/// * Fire only on a fresh entry edge (`!inside_last && inside_now`). Standing
///   still inside never re-fires; leaving + re-entering does (unless `once`).
/// * `once` ⇒ fire at most once, then disarm (`fired = true`) until the level
///   reset rebuilds `states`. `!once` ⇒ re-fire on each fresh entry.
/// * `ActivateNode` drives the resolved input node to `1` via
///   [`set_input_value`] (re-running the forward pass / re-solving). An
///   unresolved index (`usize::MAX`) or a non-input node is a no-op on the
///   puzzle — the label is still surfaced.
/// * `states.len()` must match `triggers.len()` (both indexed by trigger order);
///   a length mismatch (defensive) processes only the overlapping prefix.
///
/// `re_world::Map::triggers_containing` is NOT reused here because we need the
/// per-trigger INDEX to pair with `states`; this walks the same slice with the
/// same floor-cell containment test.
pub fn apply_triggers(
    triggers: &[Trigger],
    cam: &CameraPose,
    states: &mut [TriggerRuntime],
    puzzle: &mut PuzzleState,
) -> Vec<FiredTrigger> {
    let mut fired = Vec::new();
    if triggers.is_empty() {
        return fired;
    }
    // A non-finite pose is "inside nothing" (never fire on garbage).
    let cell = if cam.pos.is_finite() {
        Some((
            cam.pos.x.floor() as i32,
            cam.pos.y.floor() as i32,
            cam.pos.z.floor() as i32,
        ))
    } else {
        None
    };
    let n = triggers.len().min(states.len());
    for i in 0..n {
        let t = &triggers[i];
        let inside_now = matches!(cell, Some((x, y, z)) if t.contains(x, y, z));
        let entered = inside_now && !states[i].inside_last;
        states[i].inside_last = inside_now;
        if !entered {
            continue;
        }
        // Entry edge. Respect the `once` disarm latch.
        if t.once && states[i].fired {
            continue;
        }
        let label = match &t.action {
            TriggerAction::ActivateNode { name, node_index } => {
                // Drive the node like an input → re-runs the forward pass + re-solves.
                // An unresolved (`usize::MAX`) or non-input index no-ops in
                // `set_input_value`, leaving the puzzle untouched but still
                // surfacing the event.
                let _ = set_input_value(puzzle, *node_index, 1);
                name.clone()
            }
            TriggerAction::Event { label } => label.clone(),
            // `#[non_exhaustive]`: an unknown future action fires no side-effect
            // and surfaces an empty label.
            _ => String::new(),
        };
        states[i].fired = true;
        fired.push(FiredTrigger { label });
    }
    fired
}

/// Drive every `[zone_input]` floor-zone from the player's CURRENT cell: each
/// zone's input node is held at `1` while the camera stands inside the zone's
/// box, `0` otherwise. Re-solves only on a real change (via
/// [`set_input_value`], which no-ops when the value is unchanged).
///
/// Pure + headless-testable: takes the parsed zones, the camera, and a `&mut`
/// puzzle — no Bevy, no terminal. Returns `true` iff any input changed (so the
/// caller / a test can assert a re-solve happened). The camera's floored cell
/// (`pos.floor()`) is the occupancy test, matching the cell-quantized world.
pub fn apply_zone_inputs(zones: &[ZoneInput], cam: &CameraPose, puzzle: &mut PuzzleState) -> bool {
    if zones.is_empty() {
        return false;
    }
    let (cx, cy, cz) = (
        cam.pos.x.floor() as i32,
        cam.pos.y.floor() as i32,
        cam.pos.z.floor() as i32,
    );
    let mut changed = false;
    for z in zones {
        let inside = z.contains(cx, cy, cz);
        if set_input_value(puzzle, z.node_index, i32::from(inside)) {
            changed = true;
        }
    }
    changed
}
