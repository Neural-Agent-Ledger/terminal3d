//! `state_bridge` — the puzzle → render STATE BUS bridge (flow-mark/pipe-side
//! encoding onto [`EmitterOrigin`]/[`EmitterSpec`]). No seams: this crate owns
//! every puzzle semantic named in the module docs below, and only encodes the
//! resulting `u32`/`f32` onto the generic [`re_core::CellStateMap`] that
//! `re_render` reads.
//!
//! This is the bridge between two worlds: it walks the live [`PuzzleState`]
//! (which it is allowed to know about) and encodes each node's status into the
//! GENERIC [`CellStateMap`] (`tag → state:u32 / pulse:f32`) that `re_render` reads
//! WITHOUT knowing it is a puzzle. The renderer never sees a `re_puzzle` type — only
//! the `u32`/`f32` per tag — so crate purity is preserved (the same discipline the
//! `EmitterSpec`/`Emitter` split uses).
//!
//! ## Tag space
//!
//! A puzzle node's world cell carries its NODE INDEX as the `side_data` tag (see
//! `re_level::build_map`: `ObjectCell { tag: i }` for node `i`). That tag is exactly
//! what reaches the shader as the cell's `material`, and what the [`CellStateMap`] is
//! keyed by. So the bridge writes `map.set_state(node_index, encode(node))`.
//!
//! ## Encoding (the small packed enum)
//!
//! The low lane of the `u32` is the node VISUAL STATE, ordered to match the render's
//! `STATE_NODE` glyph family (member 0 = the shipped resting look):
//!
//! | value | meaning  | how                                                    |
//! |-------|----------|--------------------------------------------------------|
//! | 0     | powered  | `activation == 1` (the lit, shipped neuron look)       |
//! | 1     | dead     | `activation == 0` and the node can still matter        |
//! | 2     | decoy    | a node that can NEVER reach a door target (a red herring) |
//! | 3     | won      | a door-target node once the whole puzzle is `solved`   |
//!
//! `won` (3) wins over `powered` for door-target nodes when the level is solved, so
//! the goal nodes flare bright on the win.
//!
//! ## Pulse
//!
//! A node that JUST FIRED this rebuild (activation rose `0 → 1` since the previous
//! map) gets a one-shot `pulse = 1.0` (a brief emissive flash, the "node fired"
//! event); the bridge decays every tag's pulse toward 0 each rebuild so the
//! flash settles over a few frames. Cycle-rejection and switch-toggle sparks
//! need the event stream and are handled separately in [`event_pulse_sys`];
//! the node-fire pulse here needs only what the forward pass already makes
//! available.
//!
//! ## The solve-progress MARK (the voice materials)
//!
//! Two render materials — `re_render`'s `mnemos_hold` and `lethe_release` — are driven
//! by how far the level has come rather than by a clock. They read ONE scalar, the
//! MARK, off two reserved bus tags ([`re_render::substrate::MNEMOS_TAG`] /
//! [`LETHE_TAG`](re_render::substrate::LETHE_TAG)), quantised as parts per thousand.
//!
//! The mark is the MONOTONE HIGH-WATER of [`solve_progress`]: it rises with the
//! network and never falls, so a player can unwire their work and watch the wall
//! refuse to forget it. Only a level reset clears it — which is why it lives HERE, in
//! the per-level session state, and not in `re_render`. The renderer stays a pure
//! function of the number: scalar in, glyph out.
//!
//! It rides the `state: u32` channel and not `pulse: f32` because `pulse` is a
//! TRANSIENT kick by contract (this module zeroes it on the next rebuild); a level
//! that must be held cannot ride a channel that decays.

use bevy::prelude::*;
use re_core::{encode_flow_state, encode_pipe_side_data, CellStateMap, FlowDir};
use re_level::PipeState;
use re_puzzle::{InteractAt, PuzzleState};
use re_world::{BoldMode, EmitterOrigin, EmitterSpec};

/// Node visual-state lane values — MUST match the order of `re_render`'s
/// `glyph_family::STATE_NODE` (member 0 = the shipped resting look). These are the
/// game-side SEMANTICS the renderer stays ignorant of; it only indexes a glyph by
/// the resulting `u32`.
pub mod node_state {
    /// `activation == 1` — the lit, shipped neuron look (render member 0).
    pub const POWERED: u32 = 0;
    /// `activation == 0`, still able to matter — a cold/frozen node (member 1).
    pub const DEAD: u32 = 1;
    /// Can never reach a door target — a dim, inert decoy (member 2).
    pub const DECOY: u32 = 2;
    /// A door-target node once the puzzle is solved — a bright won pulse (member 3).
    pub const WON: u32 = 3;
}

/// How fast a one-shot fire/event PULSE decays per rebuild: the previous pulse is
/// scaled by this each time the map is rebuilt, so a `1.0` flash fades to ~0 over a
/// handful of interactions. (Rebuilds are interaction-driven, not per-frame, so this
/// is a coarse settle, which is exactly the "flash then settle" the spec wants.)
const PULSE_DECAY: f32 = 0.5;

/// The bridge's own bookkeeping: the previous node activations (to detect a `0 → 1`
/// rise → a fire pulse) and a change-detection hash so the map is only rebuilt when
/// the puzzle actually changed. A Bevy [`Resource`].
#[derive(Resource, Default)]
pub struct StateBridge {
    /// Previous-rebuild activation per node (index-aligned with `PuzzleState::nodes`).
    prev_activation: Vec<i32>,
    /// Previous-rebuild ON/OFF state per pipe (index-aligned with `PuzzleState::pipes`),
    /// so a SWITCH TOGGLE (state changed) is told apart from a CYCLE REJECT (an
    /// interaction was attempted but the pipe state did NOT change — `puzzle_interact`
    /// rolled it back). Used by [`event_pulse_sys`] to classify each event.
    prev_pipe_state: Vec<PipeState>,
    /// Previous-rebuild `solved` flag, so a DOOR-OPEN event (`false → true`) fires its
    /// flare/pulse exactly once on the solving frame.
    prev_solved: bool,
    /// A cheap fingerprint of the last puzzle status the map was built from, so an
    /// idle frame (no interaction) skips the rebuild entirely (rides the dirty-flag).
    last_fingerprint: u64,
    /// Set once a level is loaded so frame-1 builds the map even if nothing changed.
    initialised: bool,
    /// How many BASE (level-authored) emitters sit at the front of the `EmitterSet`,
    /// captured the first time [`event_pulse_sys`] runs after a (re)load. Event-burst
    /// emitters are appended AFTER these, and a compaction truncates back to this many.
    base_emitter_count: Option<usize>,
    /// Node indices that fired (`0 → 1`) in THIS rebuild, recorded by `rebuild_map`
    /// (which still has the previous activations) so [`event_pulse_sys`] — which runs
    /// AFTER the rebuild has already advanced `prev_activation` — can spawn their
    /// fire bursts. Drained each time the event system consumes it.
    fired_this_rebuild: Vec<usize>,
    /// The MONOTONE HIGH-WATER of [`solve_progress`] for this level session, in
    /// `[0,1]` — the scalar the two voice materials render.
    ///
    /// It only ever rises (`max`), so rolling a switch back does not un-fill the wall.
    /// Its lifetime is the level session: [`StateBridge::reset`] (called on teardown)
    /// zeroes it with the rest of the bookkeeping, so the next level — or the same
    /// level restarted — starts from a blank wall.
    ///
    /// `0.0` by `Default`, which is exactly the resting mark, so a bridge that has
    /// never run publishes an at-rest wall.
    solve_mark: f32,
}

impl StateBridge {
    /// Reset the bridge to its fresh-load baseline. Call on level TEARDOWN so the next
    /// level rebuilds its state + re-captures its base emitter count from scratch,
    /// even when it happens to have the same node/pipe counts as the previous one
    /// (otherwise the length-change reload heuristics would miss the transition).
    pub fn reset(&mut self) {
        *self = StateBridge::default();
    }
}

/// A cheap order-sensitive fingerprint of the puzzle's render-affecting status: the
/// activation of every node plus the solved flag. Changes iff a node's activation or
/// the win state changed — exactly when the [`CellStateMap`] must be rebuilt.
fn fingerprint(puzzle: &PuzzleState) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    };
    for n in &puzzle.nodes {
        mix(n.activation as u64);
    }
    mix(u64::from(puzzle.solved));
    mix(puzzle.nodes.len() as u64);
    // The SOLVE-PROGRESS word. Without it a switch that changes which pipes are
    // powered WITHOUT changing any node activation (an OFF pipe out of an already-cold
    // node, say) would leave the fingerprint identical, the rebuild skipped, and the
    // mark stuck one step behind the network. Mixing the quantised progress in makes
    // the skip exact: the map is rebuilt iff something the render can see changed.
    // Can only ADD rebuilds, never remove one, so the idle economy is unaffected.
    mix(u64::from(re_render::substrate::encode_mark(
        solve_progress(puzzle),
    )));
    h
}

/// How far this level has come, in `[0,1]` — POWERED PIPES over total pipes.
///
/// "Powered" is the SAME predicate `rebuild_map` uses for a pipe's flow word (ON, and
/// its source node active), reused deliberately: the wall and the pipe traces should
/// agree about what is alive, and two definitions of one word is how they would stop
/// agreeing.
///
/// Two edge cases, both deliberate:
/// * a SOLVED puzzle reads `1.0` regardless — the wall must complete when the level
///   completes, and a level whose door opens without every pipe hot would otherwise
///   stop just short;
/// * a puzzle with NO pipes reads `0.0` (or `1.0` once solved) rather than dividing by
///   zero.
///
/// Pure — no bookkeeping, no clock. The MONOTONE high-water is applied by the caller;
/// this is the instantaneous reading.
fn solve_progress(puzzle: &PuzzleState) -> f32 {
    if puzzle.solved {
        return 1.0;
    }
    if puzzle.pipes.is_empty() {
        return 0.0;
    }
    let powered = puzzle
        .pipes
        .iter()
        .filter(|p| {
            p.is_on() && p.from < puzzle.nodes.len() && puzzle.nodes[p.from].activation == 1
        })
        .count();
    powered as f32 / puzzle.pipes.len() as f32
}

/// Decide a node's visual state lane from the live puzzle. Pure; the render stays
/// ignorant of this — it only sees the returned `u32`.
fn encode_node(puzzle: &PuzzleState, idx: usize, can_reach_target: bool) -> u32 {
    let node = &puzzle.nodes[idx];
    // WON: a goal node, lit, with the whole puzzle solved → the bright flare.
    if puzzle.solved && puzzle.door_targets.contains(&idx) {
        return node_state::WON;
    }
    if node.activation == 1 {
        return node_state::POWERED;
    }
    // DECOY: a node that can never contribute to any door target — a red herring.
    // (An input or a node that can reach a target is a real, just-unpowered DEAD
    // node; only a structurally-useless node is a decoy.)
    if !can_reach_target && !node.is_input {
        return node_state::DECOY;
    }
    node_state::DEAD
}

/// For every node, whether it can reach ANY door target by following pipes forward
/// (topology only — state-independent), so the encoder can tell a genuinely
/// just-unpowered DEAD node from a structurally-useless DECOY. A door target reaches
/// itself. Cheap BFS over the small v0.1 graph.
fn nodes_reaching_targets(puzzle: &PuzzleState) -> Vec<bool> {
    let n = puzzle.nodes.len();
    let mut reach = vec![false; n];
    // Reverse adjacency: for each node, its predecessors (pipe `from` → `to`).
    let mut stack: Vec<usize> = Vec::new();
    for &t in &puzzle.door_targets {
        if t < n && !reach[t] {
            reach[t] = true;
            stack.push(t);
        }
    }
    while let Some(cur) = stack.pop() {
        for p in &puzzle.pipes {
            if p.to == cur && p.from < n && !reach[p.from] {
                reach[p.from] = true;
                stack.push(p.from);
            }
        }
    }
    reach
}

/// Rebuild `map` from `puzzle` into the [`StateBridge`] bookkeeping. Pure given the
/// inputs; factored out so it is unit-testable with no ECS. Writes node lanes +
/// fire pulses, and records the new activations for the next rise-detect.
fn rebuild_map(bridge: &mut StateBridge, puzzle: &PuzzleState, map: &mut CellStateMap) {
    let n = puzzle.nodes.len();
    if bridge.prev_activation.len() != n {
        // Level (re)loaded or first run: size the activation history, no fire pulses
        // on the very first build (every node is "new", which would flash the whole
        // board — undesirable; treat load as the baseline).
        bridge.prev_activation = vec![-1; n];
    }
    let reach = nodes_reaching_targets(puzzle);

    // Decay the previous pulses first (a flash from a prior interaction settling),
    // then set a fresh full pulse on any node that JUST fired (rose 0 → 1).
    map.clear(); // reset states; we re-derive every node's lane below.
    bridge.fired_this_rebuild.clear();
    for idx in 0..n {
        let state = encode_node(puzzle, idx, reach.get(idx).copied().unwrap_or(false));
        map.set_state(idx as u16, state);

        let act = puzzle.nodes[idx].activation;
        let prev = bridge.prev_activation[idx];
        // A 0 → 1 rise is a fire event → a full flash. A baseline (`prev == -1`,
        // a fresh load) never flashes. Otherwise carry a decayed prior pulse so an
        // earlier flash settles smoothly.
        let pulse = if prev == 0 && act == 1 {
            // Record the fire so `event_pulse_sys` (which runs after this rebuild has
            // advanced `prev_activation`) can spawn the matching burst.
            bridge.fired_this_rebuild.push(idx);
            1.0
        } else {
            // No fresh fire: keep settling whatever this tag had (the bridge's own
            // decay — `map` was cleared, so re-apply the decayed value).
            0.0
        };
        if pulse > 0.0 {
            map.set_pulse(idx as u16, pulse);
        }
        bridge.prev_activation[idx] = act;
    }

    // PIPE FLOW STATE: for every pipe, encode a
    // flow word keyed by the pipe's RENDER tag (`encode_pipe_side_data(pipe_index)`
    // — the high-bit pipe namespace the `re_level` stamper used, DISJOINT from node
    // tags). A pipe FLOWS (powered) when it is ON and its source (`from`) node is
    // active; the authored path runs input(`from`)→output(`to`), so flow scrolls
    // FORWARD along the path. An OFF or unpowered pipe encodes the cold default
    // (`FlowDir::None` → state 0), so its trace renders static/cold. The shader keys
    // the bus by the FULL flagged side_data, so this never aliases a node tag.
    for (pi, pipe) in puzzle.pipes.iter().enumerate() {
        let powered = pipe.is_on()
            && pipe.from < puzzle.nodes.len()
            && puzzle.nodes[pipe.from].activation == 1;
        let dir = if powered {
            FlowDir::Forward
        } else {
            FlowDir::None
        };
        map.set_state(encode_pipe_side_data(pi as u16), encode_flow_state(dir));
    }
    // SOLVE-PROGRESS MARK: the two voice materials read
    // how far the level has come, off two RESERVED tags. Three properties, all of
    // them living here rather than in the renderer:
    //
    //  * MONOTONE — `max`, so the mark never falls. The player can unwire their work
    //    and the wall refuses to forget it; only a reset (`StateBridge::reset`, which
    //    zeroes `solve_mark` with everything else) clears it.
    //  * RE-PUBLISHED EVERY REBUILD — `map.clear()` above zeroes the whole bus, so a
    //    mark written once would vanish on the next interaction. It is derived from
    //    the bridge's own persisted high-water, not from the map.
    //  * DISJOINT — the two tags sit at the top of the non-pipe namespace, far above
    //    the small contiguous node indices written above and below the flagged pipe
    //    tags written just now. `the_mark_tags_collide_with_no_node_or_pipe_tag`
    //    checks that against the level's real tag sets rather than by assertion.
    bridge.solve_mark = bridge.solve_mark.max(solve_progress(puzzle));
    let mark_word = re_render::substrate::encode_mark(bridge.solve_mark);
    map.set_state(re_render::substrate::MNEMOS_TAG, mark_word);
    map.set_state(re_render::substrate::LETHE_TAG, mark_word);

    // Decay note: because `map.clear()` zeros pulses each rebuild and rebuilds are
    // interaction-driven, a fire pulse shows on the interaction frame and is gone by
    // the next interaction — a crisp one-shot flash. `PULSE_DECAY` is reserved for a
    // future per-frame settle once event pulses ride a frame timer; kept here as the
    // documented decay constant.
    let _ = PULSE_DECAY;
}

/// `Update` system: rebuild the [`CellStateMap`] from the live [`PuzzleState`] when
/// the puzzle changed (an interaction / forward pass). Rides the dirty-flag — an idle
/// frame with unchanged activations skips the rebuild, so the render's existing
/// state-change-is-dirty path is unperturbed.
///
/// Ordering: runs AFTER `re_puzzle::apply_interactions` (which performs the toggle +
/// forward pass), so this frame's `CellStateMap` reflects this frame's puzzle. The
/// render's `composite_render_sys` (PostUpdate) then publishes it to the cast.
pub fn state_bridge_sys(
    puzzle: Res<PuzzleState>,
    mut bridge: ResMut<StateBridge>,
    mut map: ResMut<CellStateMap>,
) {
    let fp = fingerprint(&puzzle);
    if bridge.initialised && fp == bridge.last_fingerprint {
        return; // nothing render-affecting changed: skip (idle-frame cheap).
    }
    bridge.initialised = true;
    bridge.last_fingerprint = fp;
    rebuild_map(&mut bridge, &puzzle, &mut map);
}

// ───────────────────────── event responses ──────────────────────────
//
// The bridge turns the frame's puzzle events into (a) a transient `pulse` on the
// event cell's tag — the render already reads it as a brief emissive kick —
// and (b) a one-shot BURST emitter at the cell (a spark / flare /
// red corruption pulse). Four events, each a distinct response:
//
//   switch toggled  → a small white SPARK   + a pulse on the toggled pipe's tag
//   node fired      → (pulse already set in `rebuild_map`) + a warm FLARE at the node
//   door opened     → a bright cyan FLARE   + a pulse on every door-target node tag
//   cycle rejected  → a RED corruption BURST + a red pulse (the `while(true)` dread)

/// Max event-burst emitters appended after the level's base emitters before a
/// compaction truncates the tail back to the base set (and resets the particle pool
/// so the sticky one-shot `burst_fired` flags clear). Generous — a player toggles
/// far fewer than this between the rare compaction.
const EVENT_BURST_CAP: usize = 48;

/// Build a one-shot BURST [`EmitterSpec`] at world `cell` with `count` particles in
/// the three `colors`, sized `size`. World-anchored so it sits at the event cell.
fn burst_spec(cell: (i32, i32, i32), count: u32, colors: [u32; 3], size: f32) -> EmitterSpec {
    // NOTE: `EmitterSpec` uses `re_core`'s glam (0.33); `bevy::prelude::Vec3` is a
    // DIFFERENT glam (0.30) in this graph, so we name `re_core::Vec3` explicitly.
    use re_core::Vec3 as CoreVec3;
    let origin = EmitterOrigin::World(CoreVec3::new(
        cell.0 as f32 + 0.5,
        cell.1 as f32 + 0.5,
        cell.2 as f32 + 0.5,
    ));
    EmitterSpec::new()
        .with_origin(origin)
        .with_burst(count)
        .with_colors(colors[0], colors[1], colors[2])
        .with_size(size, 0.6)
        // A short, snappy spark: fast small particles, brief life.
        .with_direction(CoreVec3::Z, 140.0)
        .with_speed(2.4, 1.2)
        .with_bold(BoldMode::ByAge)
        .with_lifetime(0.5, 0.2)
}

/// Append a burst emitter to the live [`EmitterSet`], compacting (and resetting the
/// pool) if the event-burst tail has grown past [`EVENT_BURST_CAP`].
fn push_event_burst(
    set: &mut re_render::EmitterSet,
    pool: &mut re_render::ParticlePool,
    base_count: usize,
    spec: &EmitterSpec,
) {
    if set.emitters.len().saturating_sub(base_count) >= EVENT_BURST_CAP {
        // Truncate back to the base set and clear the pool so the new bursts at the
        // freed indices fire (the one-shot `burst_fired` flags are index-sticky).
        set.emitters.truncate(base_count);
        *pool = re_render::ParticlePool::default();
    }
    set.emitters.push(re_render::Emitter::from_spec(spec));
}

/// `Update` system (AFTER [`state_bridge_sys`]): translate this frame's puzzle
/// EVENTS into render pulses + one-shot burst emitters. Reads the `InteractAt`
/// stream with its OWN cursor (independent of `apply_interactions`), so it sees the
/// same interactions and can classify each as a successful SWITCH TOGGLE (the pipe's
/// state changed) or a CYCLE REJECT (an attempted toggle that `puzzle_interact`
/// rolled back — the pipe state is unchanged). DOOR-OPEN fires on `solved` rising.
///
/// Purity: the renderer still reads only the generic bus; this game-side system owns
/// every puzzle semantic, exactly like `rebuild_map`.
#[allow(clippy::too_many_arguments)]
pub fn event_pulse_sys(
    mut interactions: MessageReader<InteractAt>,
    puzzle: Res<PuzzleState>,
    mut bridge: ResMut<StateBridge>,
    mut map: ResMut<CellStateMap>,
    mut emitters: ResMut<re_render::EmitterSet>,
    mut pool: ResMut<re_render::ParticlePool>,
) {
    // Capture the level's base emitter count once per (re)load. `state_bridge_sys`
    // sets `initialised` on the first build; we mirror that by recording the base
    // count the first time we see a sized pipe-state vector that differs in length
    // (a fresh level) — simplest robust signal: record on the first run after the
    // pipe-state bookkeeping is (re)sized below.
    let n_pipes = puzzle.pipes.len();
    let level_reloaded = bridge.prev_pipe_state.len() != n_pipes;
    if level_reloaded {
        bridge.prev_pipe_state = puzzle.pipes.iter().map(|p| p.state).collect();
        bridge.prev_solved = puzzle.solved;
        // A fresh level: the current EmitterSet is the authored base; remember its
        // size so event bursts append after it (and compaction truncates to it).
        bridge.base_emitter_count = Some(emitters.emitters.len());
        // Drain this frame's interactions without reacting (load baseline), so a
        // stale message from the previous level can't spark on frame 1.
        interactions.clear();
        return;
    }
    let base_count = bridge.base_emitter_count.unwrap_or(emitters.emitters.len());

    // 1) INTERACTIONS → switch-toggle spark OR cycle-reject red pulse.
    for m in interactions.read() {
        let Some(pi) = puzzle.pipe_at_switch(m.x, m.y, m.z) else {
            continue; // not a switch cell — nothing to spark.
        };
        let now = puzzle.pipes[pi].state;
        let was = bridge.prev_pipe_state.get(pi).copied().unwrap_or(now);
        let cell = (m.x, m.y, m.z);
        if now != was {
            // SUCCESSFUL TOGGLE → a white spark + a pulse on the pipe's render tag.
            map.set_pulse(encode_pipe_side_data(pi as u16), 1.0);
            let spec = burst_spec(cell, 14, [0xFFFFFF, 0xCFE8FF, 0x8FB8FF], 1.0);
            push_event_burst(&mut emitters, &mut pool, base_count, &spec);
        } else {
            // ATTEMPTED BUT UNCHANGED ⇒ `puzzle_interact` rolled the move back: a
            // CYCLE REJECT (the `while(true)` premonition). A RED corruption burst +
            // a red pulse on the pipe tag.
            map.set_pulse(encode_pipe_side_data(pi as u16), 1.0);
            let spec = burst_spec(cell, 22, [0xFF2020, 0xC01010, 0x600808], 1.3);
            push_event_burst(&mut emitters, &mut pool, base_count, &spec);
        }
    }

    // 2) NODE FIRED (0 → 1 this rebuild, recorded by `rebuild_map`) → a warm flare at
    //    the node. The PULSE itself is already set by `rebuild_map`; here we add the
    //    spark. DRAIN the list so a later idle frame (no rebuild) does not re-fire it.
    let fired = std::mem::take(&mut bridge.fired_this_rebuild);
    for idx in fired {
        if let Some(cell) = node_cell(&puzzle, idx) {
            let spec = burst_spec(cell, 16, [0xFFE0A0, 0xFFB060, 0xC07020], 1.1);
            push_event_burst(&mut emitters, &mut pool, base_count, &spec);
        }
    }

    // 3) DOOR OPENED (solved rose false → true) → a bright cyan flare + a pulse on
    //    every door-target node tag (avoids the door-anchor tag-0 collision).
    if puzzle.solved && !bridge.prev_solved {
        for &t in &puzzle.door_targets {
            map.set_pulse(t as u16, 1.0);
            if let Some(cell) = node_cell(&puzzle, t) {
                let spec = burst_spec(cell, 28, [0xE0FFFF, 0x80E0FF, 0x40A0E0], 1.4);
                push_event_burst(&mut emitters, &mut pool, base_count, &spec);
            }
        }
    }

    // Record this frame's pipe states + solved for the next classification.
    for (pi, pipe) in puzzle.pipes.iter().enumerate() {
        if pi < bridge.prev_pipe_state.len() {
            bridge.prev_pipe_state[pi] = pipe.state;
        }
    }
    bridge.prev_solved = puzzle.solved;
}

/// The world cell a puzzle node lives at, recovered from any pipe path that names it
/// as an endpoint. The bridge has no direct node→cell map (the puzzle network drops
/// world coords), but a node's cell is the `from`/`to` end of one of its pipes'
/// expanded paths. `None` for an isolated node with no pipe (then no burst — the
/// pulse still kicks the node tag via the win/fire path).
fn node_cell(puzzle: &PuzzleState, idx: usize) -> Option<(i32, i32, i32)> {
    for p in &puzzle.pipes {
        if p.from == idx {
            if let Some(&c) = p.path.first() {
                return Some(c);
            }
        }
        if p.to == idx {
            if let Some(&c) = p.path.last() {
                return Some(c);
            }
        }
    }
    None
}

/// The plugin: registers the [`StateBridge`] bookkeeping + the [`CellStateMap`]
/// resource (insert-if-absent; the render plugin also inits it, first wins) and the
/// `Update` bridge systems. Added by the binary AFTER `PuzzlePlugin` so the system
/// ordering (`after(apply_interactions)`) is well-formed. The event system runs
/// AFTER the state rebuild so the pulses it sets survive into the published map.
pub struct StateBridgePlugin;

impl Plugin for StateBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StateBridge>()
            .init_resource::<CellStateMap>()
            // The event system reads the live particle resources; init-if-absent so
            // the bridge is order-independent of `EmitterPlugin` (first to init wins,
            // and `EmitterPlugin` overwrites the EmitterSet from level data anyway).
            .init_resource::<re_render::EmitterSet>()
            .init_resource::<re_render::ParticlePool>()
            .add_systems(
                Update,
                (
                    state_bridge_sys.after(re_puzzle::apply_interactions),
                    event_pulse_sys.after(state_bridge_sys),
                ),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_puzzle::{Pipe, PuzzleNode};
    // `PipeState` is already in scope via `use super::*` (the module imports it).
    use re_core::decode_flow_state;

    /// Build a tiny 3-node chain: input(0) → hidden(1) → output(2, a door target).
    fn chain_puzzle(act: [i32; 3], solved: bool) -> PuzzleState {
        let nodes = vec![
            PuzzleNode {
                is_input: true,
                activation: act[0],
                ..Default::default()
            },
            PuzzleNode {
                threshold: 1,
                activation: act[1],
                ..Default::default()
            },
            PuzzleNode {
                threshold: 1,
                is_output: true,
                activation: act[2],
                ..Default::default()
            },
        ];
        let pipes = vec![
            Pipe {
                from: 0,
                to: 1,
                state: PipeState::On,
                ..Default::default()
            },
            Pipe {
                from: 1,
                to: 2,
                state: PipeState::On,
                ..Default::default()
            },
        ];
        PuzzleState {
            nodes,
            pipes,
            inputs: vec![0; 3],
            door_targets: vec![2],
            sequence: Default::default(),
            solved,
            attempts: 0,
            ..Default::default()
        }
    }

    #[test]
    fn powered_and_dead_nodes_encode_differently() {
        // Node 0 fired, node 1 not → they must encode to DIFFERENT lanes.
        let puzzle = chain_puzzle([1, 0, 0], false);
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);
        assert_eq!(map.state(0), node_state::POWERED, "fired node = powered");
        assert_eq!(
            map.state(1),
            node_state::DEAD,
            "unfired (but reachable) node = dead"
        );
        assert_ne!(map.state(0), map.state(1), "powered ≠ dead");
    }

    #[test]
    fn solved_door_target_encodes_won() {
        let puzzle = chain_puzzle([1, 1, 1], true);
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);
        // The door target (node 2) flares WON; the others are powered.
        assert_eq!(map.state(2), node_state::WON, "solved target = won");
        assert_eq!(map.state(0), node_state::POWERED);
    }

    #[test]
    fn decoy_node_that_cannot_reach_a_target() {
        // A 2-node graph: node 0 is the door target; node 1 is a dead-end that points
        // at nothing (no pipe out, not a target) → a DECOY.
        let nodes = vec![
            PuzzleNode {
                threshold: 1,
                activation: 0,
                ..Default::default()
            },
            PuzzleNode {
                threshold: 1,
                activation: 0,
                ..Default::default()
            },
        ];
        let puzzle = PuzzleState {
            nodes,
            pipes: vec![],
            inputs: vec![0; 2],
            door_targets: vec![0],
            sequence: Default::default(),
            solved: false,
            attempts: 0,
            ..Default::default()
        };
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);
        assert_eq!(
            map.state(0),
            node_state::DEAD,
            "the (unpowered) target is dead, not decoy"
        );
        assert_eq!(
            map.state(1),
            node_state::DECOY,
            "the dead-end node is a decoy"
        );
    }

    #[test]
    fn newly_fired_node_gets_a_pulse_then_settles() {
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        // Baseline build (load): no flashes.
        let p0 = chain_puzzle([0, 0, 0], false);
        rebuild_map(&mut bridge, &p0, &mut map);
        assert_eq!(map.pulse(1), 0.0, "load baseline does not flash");
        // Node 1 fires (0 → 1): a flash on its tag.
        let p1 = chain_puzzle([1, 1, 0], false);
        rebuild_map(&mut bridge, &p1, &mut map);
        assert_eq!(map.pulse(1), 1.0, "a node that just fired flashes");
        // Next rebuild with no further rise: the flash is gone (one-shot).
        let p2 = chain_puzzle([1, 1, 0], false);
        rebuild_map(&mut bridge, &p2, &mut map);
        assert_eq!(map.pulse(1), 0.0, "the fire flash is one-shot");
    }

    #[test]
    fn empty_puzzle_leaves_an_empty_map() {
        let puzzle = PuzzleState::default();
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);
        // No nodes ⇒ every tag still reads the default.
        assert_eq!(map.state(0), 0);
        assert_eq!(map.pulse(0), 0.0);
    }

    // ───────────────────────── pipe flow + events ────────────────────────

    #[test]
    fn powered_pipe_carries_flow_state_cold_pipe_does_not() {
        // Input node 0 fires; pipe 0 (from 0) is ON → it FLOWS. Pipe 1 (from 1) is ON
        // but its source (node 1) has not fired yet → COLD (no flow).
        let puzzle = chain_puzzle([1, 0, 0], false);
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);

        let pipe0_tag = encode_pipe_side_data(0);
        let pipe1_tag = encode_pipe_side_data(1);
        assert_eq!(
            decode_flow_state(map.state(pipe0_tag)),
            FlowDir::Forward,
            "a powered pipe (ON + source active) flows forward (input→output)"
        );
        assert_eq!(
            decode_flow_state(map.state(pipe1_tag)),
            FlowDir::None,
            "an unpowered pipe (source inactive) is cold/static"
        );
        // The flow words differ — a live pipe reads differently from a dead one.
        assert_ne!(map.state(pipe0_tag), map.state(pipe1_tag));
        // Pipe tags live in the high-bit namespace, DISJOINT from the node tags
        // (0,1,2), so they never alias.
        assert!(pipe0_tag >= re_core::PIPE_TAG_FLAG);
    }

    #[test]
    fn an_off_pipe_is_cold_even_with_an_active_source() {
        // Node 0 fired, but pipe 0 is switched OFF → no flow (the signal is cut).
        let mut puzzle = chain_puzzle([1, 0, 0], false);
        puzzle.pipes[0].state = PipeState::Off;
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);
        assert_eq!(
            decode_flow_state(map.state(encode_pipe_side_data(0))),
            FlowDir::None,
            "an OFF pipe never flows, even with an active source"
        );
    }

    #[test]
    fn empty_bus_has_no_pipe_flow() {
        // A puzzle with no powered pipes leaves every pipe tag at the cold default
        // (state 0) — so a decoy/unpowered network renders byte-identically cold.
        let puzzle = chain_puzzle([0, 0, 0], false);
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);
        for pi in 0..puzzle.pipes.len() {
            assert_eq!(
                map.state(encode_pipe_side_data(pi as u16)),
                0,
                "pipe {pi} cold"
            );
        }
    }

    /// A burst spec is built World-anchored at the event cell's centre and is a
    /// one-shot (the `burst` count set, continuous rate disabled).
    #[test]
    fn burst_spec_is_world_anchored_one_shot() {
        let spec = burst_spec((3, 4, 1), 20, [0xFFFFFF, 0xAAAAAA, 0x555555], 1.0);
        assert_eq!(spec.burst, Some(20), "a one-shot burst count is set");
        match spec.origin {
            EmitterOrigin::World(p) => {
                assert!((p.x - 3.5).abs() < 1e-4 && (p.y - 4.5).abs() < 1e-4);
            }
            _ => panic!("event burst must be world-anchored"),
        }
        assert!(
            spec.is_valid(),
            "the event burst spec must validate: {spec:?}"
        );
    }

    // ── EVENTS through the real Bevy systems (apply_interactions + bridge) ──

    /// A tiny app wired like the play state's reactive seam: time + puzzle + bridge.
    fn reactive_app(puzzle: PuzzleState) -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(re_puzzle::PuzzlePlugin)
            .add_plugins(StateBridgePlugin);
        app.insert_resource(puzzle);
        app
    }

    /// The chain from `chain_puzzle` but with a SWITCH on pipe 0 at world (5,5,1) and
    /// real expanded paths, so an `InteractAt` there toggles it.
    fn chain_with_switch() -> PuzzleState {
        let mut p = chain_puzzle([1, 1, 1], true);
        p.pipes[0].path = vec![(5, 5, 1), (6, 5, 1)];
        p.pipes[0].switch_cell = Some(0);
        p.pipes[1].path = vec![(6, 5, 1), (7, 5, 1)];
        p
    }

    #[test]
    fn toggling_a_switch_sets_a_pulse_on_the_pipe_tag_via_app() {
        let mut app = reactive_app(chain_with_switch());
        app.update(); // load baseline (drains any stale interaction)
        app.world_mut().write_message(InteractAt::cell(5, 5, 1));
        app.update();
        let map = app.world().resource::<CellStateMap>();
        assert!(
            map.pulse(encode_pipe_side_data(0)) > 0.0,
            "a switch toggle must pulse the toggled pipe's render tag"
        );
        assert_eq!(
            app.world().resource::<PuzzleState>().pipes[0].state,
            PipeState::Off,
            "the toggle committed (On→Off)"
        );
    }

    /// A 2-node ring with a switchable back-edge (authored OFF). Toggling it ON would
    /// close the ring → `puzzle_interact` rejects + rolls back the move; the bridge
    /// sees an unchanged pipe state for an attempted interaction → the cycle pulse.
    fn ring_with_switchable_back_edge() -> PuzzleState {
        let nodes = vec![
            PuzzleNode {
                threshold: 1,
                activation: 0,
                ..Default::default()
            },
            PuzzleNode {
                threshold: 1,
                activation: 0,
                ..Default::default()
            },
        ];
        let pipes = vec![
            Pipe {
                from: 0,
                to: 1,
                state: PipeState::On,
                path: vec![(2, 2, 1), (3, 2, 1)],
                switch_cell: None,
            },
            Pipe {
                from: 1,
                to: 0,
                state: PipeState::Off,
                path: vec![(4, 2, 1), (5, 2, 1)],
                switch_cell: Some(0),
            },
        ];
        PuzzleState {
            nodes,
            pipes,
            inputs: vec![0, 0],
            door_targets: vec![1],
            sequence: Default::default(),
            solved: false,
            attempts: 0,
            ..Default::default()
        }
    }

    // ───────────────── the solve-progress MARK (the voice pair) ─────────────

    use re_render::substrate::{decode_mark, LETHE_TAG, MARK_PARTS, MNEMOS_TAG};

    /// Both voice tags carry the same mark; read it back as the `[0,1]` scalar the
    /// renderer will see. Panics if the two disagree — they are published together
    /// and a divergence would mean one of them was clobbered.
    fn published_mark(map: &CellStateMap) -> f32 {
        assert_eq!(
            map.state(MNEMOS_TAG),
            map.state(LETHE_TAG),
            "both voices read the same mark"
        );
        decode_mark(map.state(MNEMOS_TAG))
    }

    /// A chain of `n` pipes whose first `powered` are ON out of an active source, so
    /// `solve_progress` is exactly `powered / n`. Node 0 is the input and is lit; the
    /// downstream nodes are lit or dark to match.
    fn ladder(n: usize, powered: usize) -> PuzzleState {
        let nodes: Vec<PuzzleNode> = (0..=n)
            .map(|i| PuzzleNode {
                is_input: i == 0,
                threshold: 1,
                // A node is active iff every pipe upstream of it is powered.
                activation: i32::from(i <= powered),
                ..Default::default()
            })
            .collect();
        let pipes: Vec<Pipe> = (0..n)
            .map(|i| Pipe {
                from: i,
                to: i + 1,
                state: if i < powered {
                    PipeState::On
                } else {
                    PipeState::Off
                },
                ..Default::default()
            })
            .collect();
        PuzzleState {
            nodes,
            pipes,
            inputs: vec![0; n + 1],
            door_targets: vec![n],
            sequence: Default::default(),
            solved: false,
            attempts: 0,
            ..Default::default()
        }
    }

    /// The published word is PARTS PER THOUSAND, pinned as literal numbers.
    ///
    /// Deliberately not `assert_eq!(word, encode_mark(x))` — that would only prove
    /// the bridge and the codec share a bug. The wire values are the contract between
    /// this crate and `re_render`, so they are spelled out.
    #[test]
    fn the_published_mark_is_parts_per_thousand() {
        for (n, powered, want) in [
            (4usize, 0usize, 0u32),
            (4, 1, 250),
            (4, 2, 500),
            (4, 4, 1000),
        ] {
            let mut bridge = StateBridge::default();
            let mut map = CellStateMap::new();
            rebuild_map(&mut bridge, &ladder(n, powered), &mut map);
            assert_eq!(
                map.state(MNEMOS_TAG),
                want,
                "{powered}/{n} powered pipes must publish {want} parts"
            );
            assert_eq!(
                map.state(LETHE_TAG),
                want,
                "both voices carry the same word"
            );
        }
        assert_eq!(
            MARK_PARTS, 1000,
            "the wire quantisation is parts per thousand"
        );
    }

    /// **MNEMOS keeps.** The mark is a HIGH-WATER: a player who rolls their wiring
    /// back watches the wall refuse to forget it — irreversibility lives in the
    /// mark, not the instantaneous progress.
    #[test]
    fn the_mark_never_falls_within_a_session() {
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();

        rebuild_map(&mut bridge, &ladder(4, 3), &mut map);
        let high = published_mark(&map);
        assert!((high - 0.75).abs() < 1e-6, "3 of 4 pipes powered = {high}");

        // The player unwires two switches: the instantaneous progress FALLS…
        let rolled_back = ladder(4, 1);
        assert!(
            solve_progress(&rolled_back) < high,
            "the fixture must actually roll the network back, or the test is vacuous"
        );
        // …and the published mark does not.
        rebuild_map(&mut bridge, &rolled_back, &mut map);
        assert_eq!(
            published_mark(&map),
            high,
            "a held cell must not release when the network falls back"
        );
        // Not merely sticky — it still RISES past the previous high.
        rebuild_map(&mut bridge, &ladder(4, 4), &mut map);
        assert_eq!(published_mark(&map), 1.0, "the mark must still rise");
    }

    /// A level RESET clears the wall. The mark's lifetime is the level session, which
    /// is the whole reason it lives in the bridge rather than in `re_render`.
    #[test]
    fn a_reset_clears_the_mark() {
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &ladder(4, 4), &mut map);
        assert_eq!(published_mark(&map), 1.0);

        bridge.reset();
        rebuild_map(&mut bridge, &ladder(4, 0), &mut map);
        assert_eq!(
            published_mark(&map),
            0.0,
            "a level reset starts from a blank wall"
        );

        // A mark left on the bus by a PREVIOUS level does not survive into the next
        // one. Note the mechanism: `rebuild_map`'s `map.clear()` zeroes the tag, and
        // the publish then writes the fresh 0 over it. Because a zero mark and an
        // unwritten tag both READ as 0, this assertion cannot distinguish them — it is
        // a regression check on the clear-then-republish pair as a whole, not on the
        // publish alone. (A mutation that skipped the write when the mark is 0 is
        // therefore an equivalent mutant, and was confirmed as one.)
        let mut fresh = StateBridge::default();
        let mut stale = CellStateMap::new();
        stale.set_state(MNEMOS_TAG, 999);
        stale.set_state(LETHE_TAG, 999);
        rebuild_map(&mut fresh, &ladder(4, 0), &mut stale);
        assert_eq!(
            published_mark(&stale),
            0.0,
            "a previous level's mark must not survive"
        );
    }

    /// A SOLVED puzzle publishes a full mark even if some pipe is still cold — the
    /// wall must complete when the level completes.
    #[test]
    fn a_solved_puzzle_publishes_a_full_mark() {
        let mut solved = ladder(4, 1);
        solved.solved = true;
        assert!(
            solve_progress(&ladder(4, 1)) < 1.0,
            "the same network unsolved must be short of full, or this proves nothing"
        );
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &solved, &mut map);
        assert_eq!(map.state(MNEMOS_TAG), MARK_PARTS);
    }

    /// **THE TAG COLLISION, checked against the real tag sets.** `state` is one `u16`
    /// keyspace: node lanes are written at small contiguous indices, pipe flow words at
    /// flagged high ids, and the mark at two reserved tags. If the mark landed on a
    /// node tag, that node's glyph family would index by a progress word; if a node tag
    /// landed on the mark, the wall would read a family index as progress.
    ///
    /// Measured, not asserted: build a map, then check the reserved tags against every
    /// tag the rest of this function actually wrote, and check that each of those still
    /// reads what it is supposed to while the mark is non-zero.
    #[test]
    fn the_mark_tags_collide_with_no_node_or_pipe_tag() {
        let puzzle = ladder(4, 4);
        let mut bridge = StateBridge::default();
        let mut map = CellStateMap::new();
        rebuild_map(&mut bridge, &puzzle, &mut map);
        assert!(
            map.state(MNEMOS_TAG) > 0,
            "the mark must be non-zero, or this is vacuous"
        );

        for idx in 0..puzzle.nodes.len() {
            let tag = idx as u16;
            assert!(
                tag != MNEMOS_TAG && tag != LETHE_TAG,
                "node tag {tag} hits a voice tag"
            );
            // …and the lane still reads as a lane, not as a mark.
            assert!(
                map.state(tag) <= node_state::WON,
                "node tag {tag} carries {} — a mark leaked into a glyph-family index",
                map.state(tag)
            );
        }
        for pi in 0..puzzle.pipes.len() {
            let tag = encode_pipe_side_data(pi as u16);
            assert!(
                tag != MNEMOS_TAG && tag != LETHE_TAG,
                "pipe tag {tag} hits a voice tag"
            );
            assert_eq!(
                decode_flow_state(map.state(tag)),
                FlowDir::Forward,
                "pipe {pi} must still decode as a powered flow"
            );
        }
        // The voice tags must never take a PULSE either: `pulse` is the transient
        // event kick, and an emissive flash on a progress wall would read as an event
        // that did not happen.
        assert_eq!(map.pulse(MNEMOS_TAG), 0.0);
        assert_eq!(map.pulse(LETHE_TAG), 0.0);
    }

    /// The rebuild SKIP must not strand the mark. `state_bridge_sys` returns early
    /// when the fingerprint is unchanged, so anything the render can see has to be in
    /// the fingerprint — and progress is not implied by the activations.
    ///
    /// The witness: node 2 needs BOTH its inputs, so switching one of its pipes ON
    /// powers a pipe without lighting any node. Every term the fingerprint carried
    /// before (activations, `solved`, node count) is identical across this change; the
    /// progress is not.
    #[test]
    fn a_pipe_that_powers_without_lighting_a_node_still_changes_the_fingerprint() {
        let build = |second_on: bool| {
            let nodes = vec![
                PuzzleNode {
                    is_input: true,
                    activation: 1,
                    ..Default::default()
                },
                PuzzleNode {
                    is_input: true,
                    activation: 1,
                    ..Default::default()
                },
                // Needs both inputs: one pipe on is not enough to fire it.
                PuzzleNode {
                    threshold: 2,
                    activation: 0,
                    ..Default::default()
                },
            ];
            let pipes = vec![
                Pipe {
                    from: 0,
                    to: 2,
                    state: PipeState::On,
                    ..Default::default()
                },
                Pipe {
                    from: 1,
                    to: 2,
                    state: if second_on {
                        PipeState::On
                    } else {
                        PipeState::Off
                    },
                    ..Default::default()
                },
            ];
            PuzzleState {
                nodes,
                pipes,
                inputs: vec![1, 1, 0],
                door_targets: vec![2],
                sequence: Default::default(),
                solved: false,
                attempts: 0,
                ..Default::default()
            }
        };
        let (before, after) = (build(false), build(true));
        // The witness is only a witness if NOTHING else moved.
        assert_eq!(
            before
                .nodes
                .iter()
                .map(|n| n.activation)
                .collect::<Vec<_>>(),
            after.nodes.iter().map(|n| n.activation).collect::<Vec<_>>(),
            "the fixture must leave every activation alone"
        );
        assert_eq!(before.solved, after.solved);
        assert!(
            solve_progress(&after) > solve_progress(&before),
            "the fixture must actually raise the progress ({} → {})",
            solve_progress(&before),
            solve_progress(&after)
        );
        assert_ne!(
            fingerprint(&before),
            fingerprint(&after),
            "a change the voice walls can see must not be skipped as an idle frame"
        );
    }

    /// Through the REAL systems: a switch toggle that powers the network raises the
    /// published mark, and toggling it back does not lower it.
    ///
    /// The unit tests above call `rebuild_map` directly; this one goes through
    /// `state_bridge_sys`, including its fingerprint SKIP — which is the part that
    /// could silently strand the mark one interaction behind the network.
    #[test]
    fn the_mark_rises_through_the_real_bridge_system_and_does_not_fall() {
        // Pipe 0 is authored ON with an active source, so the level loads at 1/2.
        let mut app = reactive_app(chain_with_switch());
        app.update();
        let after_load = published_mark(app.world().resource::<CellStateMap>());
        assert!(
            after_load > 0.0,
            "the loaded network is partly powered ({after_load})"
        );

        // Flip pipe 0 OFF: the instantaneous progress falls, the mark must not.
        app.world_mut().write_message(InteractAt::cell(5, 5, 1));
        app.update();
        assert_eq!(
            app.world().resource::<PuzzleState>().pipes[0].state,
            PipeState::Off,
            "the toggle committed"
        );
        let now = solve_progress(app.world().resource::<PuzzleState>());
        assert!(
            now < after_load,
            "the toggle must actually cut power, or this is vacuous"
        );
        assert_eq!(
            published_mark(app.world().resource::<CellStateMap>()),
            after_load,
            "the mark fell when the player unwired the network"
        );

        // Flip it back on: still at the high-water, never above what was reached.
        app.world_mut().write_message(InteractAt::cell(5, 5, 1));
        app.update();
        assert_eq!(
            published_mark(app.world().resource::<CellStateMap>()),
            after_load
        );
    }

    #[test]
    fn a_cycle_rejected_toggle_sets_a_pulse_via_app() {
        let mut app = reactive_app(ring_with_switchable_back_edge());
        app.update(); // load baseline
        app.world_mut().write_message(InteractAt::cell(4, 2, 1));
        app.update();
        assert_eq!(
            app.world().resource::<PuzzleState>().pipes[1].state,
            PipeState::Off,
            "the cycle toggle was rolled back"
        );
        let map = app.world().resource::<CellStateMap>();
        assert!(
            map.pulse(encode_pipe_side_data(1)) > 0.0,
            "a cycle-rejected toggle must pulse the pipe tag (the while(true) premonition)"
        );
    }
}
