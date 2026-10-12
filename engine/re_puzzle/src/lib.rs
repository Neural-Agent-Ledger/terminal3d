#![warn(missing_docs)]
//! `re_puzzle` — the binary neural-network puzzle runtime.
//!
//! A Rust port of legacy `src/puzzle.c` + `src/network.c`. The puzzle is a
//! **binary neural network**: the player toggles pipe-switches, activation
//! propagates through threshold nodes (a Kahn topological forward pass), and
//! the level is **won** when every door-target node activates.
//!
//! Two layers, deliberately separated:
//!
//! * **Network math** ([`PuzzleState`], [`network_forward`], [`puzzle_interact`],
//!   [`check_win`]) — plain structs, no Bevy, no terminal. This is the faithful
//!   port of the C forward pass + interaction model. It compiles and tests with
//!   no engine present.
//! * **Bevy plugin** ([`PuzzlePlugin`], [`InteractAt`], [`Solved`]) — the thin
//!   ECS wiring that drives the network from interaction messages and exposes
//!   the solved state as a resource.
//!
//! The network is built from a parsed [`re_level::Level`] via
//! [`PuzzleState::from_level`] (or [`spawn_from_level`] for the ECS path).
//!
//! Dependency direction matches the engine contract: this crate knows nothing
//! about notcurses / rendering / input. It depends only on `re_level` (the data
//! shapes) and `bevy` (the ECS plugin seam).

use bevy::prelude::*;

use re_level::{Level, PipeState};

/// An `(x, y, z)` integer cell coordinate (matches `re_level`'s `Coord3`).
pub type Coord3 = (i32, i32, i32);

/// Hard cap on nodes per network, mirroring legacy `NETWORK_MAX_NODES`. The
/// forward pass uses this only as a defensive bound; the level loader keeps
/// shipped levels well under it.
pub const NETWORK_MAX_NODES: usize = 64;

/// Why a [`network_forward`] pass failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardError {
    /// The pipe graph is not a DAG — a cycle was detected (fewer nodes were
    /// topologically processed than exist). Legacy returns `-1` for this.
    ///
    /// NOTE on the L12 finale: this error is what the ordinary levels treat as
    /// "unsolvable, roll the move back". When a level sets
    /// [`PuzzleState::cycle_keep`] the forward pass does NOT raise this — the
    /// detected cycle is instead *kept* as a settled terminal state (see
    /// [`network_forward`]), so the loop the player wired stands. `Cycle` is
    /// therefore only ever produced by a `cycle_keep == false` network.
    Cycle,
    /// A pipe references a node index outside `[0, n_nodes)`. Legacy returns
    /// `-1` for this (malformed graph).
    BadEdge,
    /// The network exceeds [`NETWORK_MAX_NODES`].
    TooManyNodes,
}

/// One node ("neuron") in the binary network.
///
/// Port of legacy `Node` (the puzzle-relevant fields only — render-only fields
/// like word pools and colors stay in `re_level::Node`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PuzzleNode {
    /// A non-input node fires when the sum of active incoming ON-pipe source
    /// activations is `>= threshold`. Ignored for input nodes.
    pub threshold: i32,
    /// Driven by the level's `[inputs]` pattern, not by pipes.
    pub is_input: bool,
    /// Editor flag; emit clauses / door targets may name any node.
    pub is_output: bool,
    /// The cached forward-pass result (`0` / `1`).
    pub activation: i32,
}

/// One pipe (directed edge) in the network.
///
/// Port of legacy `Pipe`, trimmed to the puzzle-relevant fields.
#[derive(Clone, Debug, Default)]
pub struct Pipe {
    /// Source node index into [`PuzzleState::nodes`].
    pub from: usize,
    /// Destination node index into [`PuzzleState::nodes`].
    pub to: usize,
    /// `On` carries the source activation; `Off` contributes `0`.
    pub state: PipeState,
    /// The expanded contiguous cell trace (from `re_level`).
    pub path: Vec<Coord3>,
    /// Index into `path` of the interactive switch cell, or `None` for a
    /// switchless pipe. The world coordinate the player interacts with is
    /// `path[switch_cell]`.
    pub switch_cell: Option<usize>,
}

impl Pipe {
    /// The world cell the player toggles this pipe by interacting with, if any.
    #[inline]
    pub fn switch_coord(&self) -> Option<Coord3> {
        self.switch_cell.and_then(|i| self.path.get(i).copied())
    }

    /// True iff the pipe is currently carrying activation.
    #[inline]
    pub fn is_on(&self) -> bool {
        self.state == PipeState::On
    }
}

/// One ordered target output-state in a [`SequenceLock`] (the L11 mechanic).
///
/// A pattern is a constraint over a set of **observed output nodes**: each
/// `(node, want_on)` pair says that node must be firing (`want_on == true`) or
/// dark (`false`) for this pattern to be considered "matched". The set of
/// observed nodes is the SAME across every pattern in a lock (the level
/// declares one fixed readout window); the patterns differ in which of those
/// nodes are required ON — that is what makes them **distinct output states**.
///
/// A pattern with an empty `bits` never matches (a degenerate pattern can't
/// lock — mirrors `check_win`'s empty-target rule).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeqPattern {
    /// `(node_index, want_on)` constraints. ALL must hold for a match.
    pub bits: Vec<(usize, bool)>,
}

impl SeqPattern {
    /// True iff every constraint in this pattern holds against the current node
    /// activations. An out-of-range node index, or an empty pattern, never
    /// matches (defensive — a malformed pattern can never lock the puzzle).
    #[inline]
    pub fn matches(&self, nodes: &[PuzzleNode]) -> bool {
        if self.bits.is_empty() {
            return false;
        }
        self.bits.iter().all(|&(idx, want_on)| {
            nodes
                .get(idx)
                .is_some_and(|n| (n.activation == 1) == want_on)
        })
    }
}

/// The opt-in **sequence-lock** win mode (L11 "Inner Circuit").
///
/// Default (`patterns` empty) ⇒ the puzzle uses the ordinary single-target
/// [`check_win`] path, byte-identical to a level with no lock declared.
///
/// When `patterns` is non-empty the win condition becomes an ORDERED state
/// machine: the output must be driven to match `patterns[0]`, then
/// `patterns[1]`, … in order. Each forward pass, if the CURRENT (not-yet-locked)
/// pattern matches, `progress` advances by one — a **lock**. The puzzle is won
/// when `progress == patterns.len()`.
///
/// **Anti-suffocation (the design contract):** locks are monotone. A pattern
/// that fails to match on a given pass does NOT reset earlier locks — it simply
/// doesn't advance, so the player keeps re-trying *only the current* pattern.
/// The whole sequence is never thrown away; the strain is attention, not
/// tedium. Once won, `progress` stays at `patterns.len()` (a solved lock holds).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceLock {
    /// The ordered target output-states. Empty ⇒ inactive (single-target win).
    pub patterns: Vec<SeqPattern>,
    /// How many patterns are locked (`0..=patterns.len()`). `== patterns.len()`
    /// means the whole sequence is solved.
    pub progress: usize,
}

impl SequenceLock {
    /// True iff this lock is active (the level declared an ordered sequence).
    #[inline]
    pub fn is_active(&self) -> bool {
        !self.patterns.is_empty()
    }

    /// True iff every pattern is locked (`progress == patterns.len()`). An
    /// inactive lock (no patterns) is never "won" by this — the caller falls
    /// back to [`check_win`].
    #[inline]
    pub fn is_won(&self) -> bool {
        self.is_active() && self.progress >= self.patterns.len()
    }

    /// Advance the lock against the current node activations, then report
    /// whether the whole sequence is now solved.
    ///
    /// At most ONE lock advances per call (the current pattern). Earlier locks
    /// are never revisited (monotone — anti-suffocation). A non-matching current
    /// pattern leaves `progress` untouched. Returns [`SequenceLock::is_won`].
    fn advance(&mut self, nodes: &[PuzzleNode]) -> bool {
        if !self.is_active() {
            return false;
        }
        if self.progress < self.patterns.len() && self.patterns[self.progress].matches(nodes) {
            self.progress += 1;
        }
        self.is_won()
    }

    /// Reset progress back to zero (a full puzzle reset). The patterns
    /// themselves (the authored sequence) are kept.
    fn reset_progress(&mut self) {
        self.progress = 0;
    }
}

/// The whole binary network plus its puzzle bookkeeping.
///
/// This is the engine-agnostic heart of the crate: pure data + the forward
/// pass. It is wrapped in a Bevy [`Resource`] for the plugin path, but every
/// method works standalone (no `World`, no `Commands`).
#[derive(Resource, Clone, Debug, Default)]
pub struct PuzzleState {
    /// The neurons (parallel to `re_level::Level::nodes`, same order).
    pub nodes: Vec<PuzzleNode>,
    /// The directed edges with toggleable switches.
    pub pipes: Vec<Pipe>,
    /// Length `nodes.len()`; the fixed `0/1` input pattern driving the
    /// `is_input` nodes (resolved from the level's `[inputs]` by node name).
    pub inputs: Vec<i32>,
    /// Node indices that must all activate to win.
    pub door_targets: Vec<usize>,
    /// The opt-in **inverted** targets (L12 `exit 0`). Node indices that must
    /// all read **0** (NOT fire) to win, in ADDITION to every `door_targets`
    /// node firing. Empty by default ⇒ the ordinary win is byte-identical (a
    /// level with no `target_off` behaves exactly as before).
    pub target_off: Vec<usize>,
    /// The opt-in [`SequenceLock`] (L11). Empty by default ⇒ the ordinary
    /// single-target win; when populated it OVERRIDES the win check with the
    /// ordered-pattern state machine. Always additive: a level that declares
    /// no `[sequence]` leaves this default and behaves exactly as before.
    pub sequence: SequenceLock,
    /// The opt-in **cycle-keep** flag (L12 `while(true)`). Default `false` ⇒ a
    /// detected cycle is the ordinary [`ForwardError::Cycle`] (the move rolls
    /// back). When `true`, [`network_forward`] does NOT error on a cycle: the
    /// cyclic nodes latch to a deterministic settled state and the forward
    /// pass terminates normally (see [`network_forward`]).
    pub cycle_keep: bool,
    /// Runtime flag set by [`network_forward`]: `true` iff the most recent
    /// pass settled a KEPT cycle (only possible when `cycle_keep`). It is the
    /// `while(true)` win signal — a wired loop that stood.
    pub cycle_held: bool,
    /// The cap on simultaneously-`On` pipes. `0` = UNLIMITED. Copied verbatim
    /// from [`re_level::Level::power_budget`] by [`PuzzleState::from_level`].
    pub power_budget: u32,
    /// Cached result of the last win check ([`check_win`], or the sequence
    /// lock / inverted / cycle-keep modes when active).
    pub solved: bool,
    /// Player-initiated toggles since construction.
    pub attempts: u32,
}

impl PuzzleState {
    /// Build the puzzle network from a parsed [`Level`].
    ///
    /// Ports the load-time half of legacy `puzzle_init`: copies the node /
    /// pipe / input / door-target data out of the level, then runs an initial
    /// [`network_forward`] + [`check_win`]. On a cycle / malformed graph the
    /// forward pass error is propagated (the legacy loader treats a non-DAG as
    /// a level-load error).
    ///
    /// `inputs` is resolved by NODE NAME: `re_level` stores `[inputs]` as
    /// `(name, value)` pairs, so each pair is mapped to its node index. Names
    /// that do not resolve are ignored (the parser already validated them).
    pub fn from_level(level: &Level) -> Result<Self, ForwardError> {
        let nodes: Vec<PuzzleNode> = level
            .nodes
            .iter()
            .map(|n| PuzzleNode {
                threshold: n.threshold,
                is_input: n.is_input,
                is_output: n.is_output,
                activation: 0,
            })
            .collect();

        let pipes: Vec<Pipe> = level
            .pipes
            .iter()
            .map(|p| Pipe {
                from: p.from_node,
                to: p.to_node,
                state: p.state,
                path: p.path.clone(),
                switch_cell: p.switch_cell,
            })
            .collect();

        // Resolve [inputs] (name, value) → a per-node 0/1 vector.
        let mut inputs = vec![0i32; nodes.len()];
        for (name, value) in &level.inputs {
            if let Some(idx) = level.node_index(name) {
                inputs[idx] = if *value != 0 { 1 } else { 0 };
            }
        }

        // Resolve the opt-in [sequence] lock (L11). Each authored pattern names
        // output nodes that must be ON (and, implicitly, every OTHER observed
        // node OFF) at that step; `re_level` already resolved names → indices
        // and recorded the fixed observation window. Empty ⇒ inactive (the
        // ordinary single-target win, byte-identical to a level with no lock).
        let sequence = SequenceLock {
            patterns: level
                .sequence
                .patterns
                .iter()
                .map(|p| SeqPattern {
                    bits: p.bits.clone(),
                })
                .collect(),
            progress: 0,
        };

        let mut state = PuzzleState {
            nodes,
            pipes,
            inputs,
            door_targets: level.door.target_parts.clone(),
            // Opt-in INVERTED targets (L12 `exit 0`): nodes that must read 0.
            // Empty by default ⇒ the ordinary win is byte-identical.
            target_off: level.door.target_off.clone(),
            sequence,
            // Opt-in CYCLE-KEEP (L12 `while(true)`): let a wired loop stand
            // instead of rolling back. Default false ⇒ today's cycle-rollback.
            cycle_keep: level.cycle_keep,
            cycle_held: false,
            // Opt-in POWER BUDGET: at most N pipes ON at once. `0` (absent key)
            // ⇒ unlimited, i.e. every pre-budget level is byte-identical.
            power_budget: level.power_budget,
            solved: false,
            attempts: 0,
        };

        network_forward(&mut state)?;
        state.refresh_solved();
        Ok(state)
    }

    /// Re-evaluate the win flag from the current node activations, advancing the
    /// [`SequenceLock`] if one is active.
    ///
    /// This is the SINGLE win-evaluation seam every runtime mutation path goes
    /// through ([`PuzzleState::from_level`], [`puzzle_interact`],
    /// [`set_input_value`], [`PuzzleState::reset_to`]). It is the only place the
    /// sequence-lock semantics live:
    ///
    /// * **No sequence declared** (`sequence` inactive) ⇒ exactly
    ///   `self.solved = check_win(self, &self.door_targets)`. This is the
    ///   ordinary single-target path — byte-identical to before the lock
    ///   existed. (Existing tests that set `state.solved = check_win(...)`
    ///   directly stay correct because that is what this reduces to.)
    /// * **Sequence active** ⇒ advance the ordered lock against the current
    ///   activations (monotone, anti-suffocation) and set `solved` from
    ///   [`SequenceLock::is_won`]. `door_targets` / [`check_win`] are NOT
    ///   consulted for the win flag in this mode (the lock owns the win).
    ///
    /// Pure function of the already-computed `activation` cache — it never runs
    /// a forward pass itself; the caller does that first.
    ///
    /// L12 finale extensions (both opt-in, both additive):
    ///
    /// * **`while(true)` (cycle-keep).** If the last forward pass settled a kept
    ///   cycle (`cycle_held`, only possible when [`cycle_keep`](Self::cycle_keep)
    ///   is set), the player has wired the loop that stands ⇒ won. This is
    ///   checked FIRST after the sequence path because the held cycle is itself a
    ///   terminal ending — the network reached a state the ordinary topological
    ///   `check_win` cannot describe (the door targets may never even be reached).
    /// * **`exit 0` (inverted target).** When no cycle is held, the win is the
    ///   combined check ([`check_win_inverted`]): every `door_targets` node fires
    ///   AND every `target_off` node does NOT. With an empty `target_off` (the
    ///   default) this reduces EXACTLY to `check_win(self, &self.door_targets)` —
    ///   byte-identical to before the feature existed.
    pub fn refresh_solved(&mut self) {
        if self.sequence.is_active() {
            self.solved = self.sequence.advance(&self.nodes);
        } else if self.cycle_held {
            // A kept cycle stands: the `while(true)` ending. (Only reachable when
            // `cycle_keep` is set, since `cycle_held` is otherwise never raised.)
            self.solved = true;
        } else {
            self.solved = check_win_inverted(self, &self.door_targets, &self.target_off);
        }
    }

    // ── power budget ────────────────────────────────────────────────────────

    /// How many pipes are currently `On` — the level's **power draw**.
    ///
    /// This counts EVERY pipe, not only the switchable ones. A level may author
    /// a permanently-`On` switchless pipe (L5's `present->hidden`, driven by a
    /// floor zone rather than a switch); that wire is genuinely carrying power
    /// and so genuinely occupies a unit of the budget. Counting it is what makes
    /// the HUD's `power N/M` readout honest — the number the player sees is the
    /// number the rule enforces, with no invisible exemptions.
    #[inline]
    pub fn power_used(&self) -> u32 {
        self.pipes.iter().filter(|p| p.is_on()).count() as u32
    }

    /// True iff this level declares a finite power budget (`power_budget > 0`).
    /// A `0` budget means UNLIMITED, not "no power" — that is the absent-key
    /// default and it must stay indistinguishable from the pre-rule behaviour.
    #[inline]
    pub fn has_power_budget(&self) -> bool {
        self.power_budget > 0
    }

    /// Could one more pipe be switched `On` right now without breaking the
    /// budget? Always `true` when the budget is unlimited.
    #[inline]
    pub fn power_available(&self) -> bool {
        !self.has_power_budget() || self.power_used() < self.power_budget
    }

    /// Find the pipe whose switch cell maps to world `(x, y, z)`. Port of
    /// legacy `network_pipe_at_switch_cell` — linear scan, fine for v0.1.
    pub fn pipe_at_switch(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        self.pipes
            .iter()
            .position(|p| p.switch_coord() == Some((x, y, z)))
    }

    /// Snapshot the current pipe states (for reset). Mirrors the
    /// `initial_pipe_states` snapshot legacy `puzzle_init` records.
    pub fn pipe_states(&self) -> Vec<PipeState> {
        self.pipes.iter().map(|p| p.state).collect()
    }

    /// The indices of every node that fired (`activation == 1`) in the SAME
    /// **forward-pass topological order** [`network_forward`] processes them
    /// (inputs → hidden → output). This is the deterministic order the host
    /// game's word→story mechanic assembles a level's "memory readout" phrase
    /// in: the phrase literally retraces the signal path through the network.
    ///
    /// Re-derives the Kahn order from the current pipe topology (state-
    /// independent in-degree, exactly as `network_forward` does) and reads the
    /// already-computed `activation` cache — it does NOT re-run the forward pass
    /// or mutate anything. If the graph is cyclic (no valid topo order) the
    /// nodes that never reached in-degree zero are appended in index order, so
    /// the result is still a stable, fired-only list (used by the denial path).
    pub fn fired_in_topo_order(&self) -> Vec<usize> {
        let order = self.topo_order();
        order
            .into_iter()
            .filter(|&i| self.nodes.get(i).is_some_and(|n| n.activation == 1))
            .collect()
    }

    /// The Kahn topological processing order of every node, mirroring the seed +
    /// dequeue order of [`network_forward`]. Cyclic remainder nodes (which never
    /// reach in-degree zero) are appended in ascending index order so the order
    /// always covers all nodes. Pure read of the pipe topology — no mutation.
    fn topo_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_deg = vec![0i32; n];
        for p in &self.pipes {
            if p.to < n && p.from < n {
                in_deg[p.to] += 1;
            }
        }
        let mut queue: Vec<usize> = Vec::with_capacity(n);
        let mut enqueued = vec![false; n];
        for i in 0..n {
            if in_deg[i] == 0 || self.nodes[i].is_input {
                queue.push(i);
                enqueued[i] = true;
            }
        }
        let mut head = 0usize;
        let mut order: Vec<usize> = Vec::with_capacity(n);
        while head < queue.len() {
            let idx = queue[head];
            head += 1;
            order.push(idx);
            for p in &self.pipes {
                if p.from != idx {
                    continue;
                }
                let to = p.to;
                if to >= n {
                    continue;
                }
                in_deg[to] -= 1;
                if in_deg[to] <= 0 && !enqueued[to] {
                    queue.push(to);
                    enqueued[to] = true;
                }
            }
        }
        // Append any cyclic remainder (nodes never enqueued) in index order.
        for (i, &done) in enqueued.iter().enumerate() {
            if !done {
                order.push(i);
            }
        }
        order
    }

    /// Restore pipe states from a snapshot, re-run the forward pass, re-check
    /// win, and count the attempt. Port of legacy `puzzle_reset`. A cycle on
    /// restore is swallowed (defensive — impossible if construction succeeded).
    pub fn reset_to(&mut self, snapshot: &[PipeState]) {
        for (pipe, &s) in self.pipes.iter_mut().zip(snapshot) {
            pipe.state = s;
        }
        // A full reset rewinds the sequence-lock progress as well: the player is
        // back at the start configuration, so they must re-drive P1→…→Pn. (For a
        // non-sequence level this is a no-op.) Reset progress BEFORE re-checking
        // so `refresh_solved` re-locks from zero against the restored state.
        self.sequence.reset_progress();
        // `network_forward` re-derives `cycle_held` from scratch each pass, so a
        // reset that restores a non-cyclic config clears any prior kept cycle.
        let _ = network_forward(self);
        self.refresh_solved();
        self.attempts += 1;
    }
}

/// Run a full forward pass over `state`, setting `activation` on every node.
///
/// Faithful port of legacy `network_forward` (Kahn topological sort):
///
/// 1. In-degree is computed from PIPE TOPOLOGY (state-independent): a pipe
///    `a -> b` adds one to `in_deg[b]` regardless of its on/off state, so the
///    evaluation order is stable as the player toggles pipes.
/// 2. Seed the work queue with every node whose `in_deg == 0` OR that is an
///    input (inputs are sources by construction).
/// 3. Pop a node and set its activation:
///    * input node → `inputs[idx] != 0`.
///    * other node → sum the source activations of INCOMING pipes that are ON,
///      fire iff `sum >= threshold`.
///
///    Then decrement every successor's in-degree, enqueuing on zero.
/// 4. If fewer than `n_nodes` were processed, the rest form a cycle → error.
///
/// Returns `Ok(())` on success, or a [`ForwardError`] on a malformed graph,
/// a cycle, or a network larger than [`NETWORK_MAX_NODES`].
///
/// Cost: `O(n_nodes + n_pipes)` (the inner activation/successor scans walk all
/// pipes, matching the C implementation's structure for v0.1 graph sizes).
///
/// **L12 cycle-keep edge model.** When [`PuzzleState::cycle_keep`] is set, the
/// in-degree (Step 1) and the successor decrement (Step 3) count only the pipes
/// that are currently **ON**. This is what makes a wired loop something the
/// player can OPEN and CLOSE by toggling: a ring whose closing pipe is OFF is
/// acyclic (it evaluates as an ordinary chain → the `exit 0` branch is
/// reachable), and turning that pipe ON closes the ring (→ the kept cycle, the
/// `while(true)` branch). The DEFAULT (`cycle_keep == false`) keeps the
/// historical **state-independent** in-degree exactly — a topological loop is a
/// cycle whether its pipes are on or off — so every pre-L12 level is
/// byte-identical. The ON-only model is reachable ONLY through the opt-in flag.
pub fn network_forward(state: &mut PuzzleState) -> Result<(), ForwardError> {
    // Every pass re-derives whether a cycle was kept; clear the prior flag first
    // so `cycle_held` is always a property of the CURRENT topology (a reset or a
    // toggle that opens the loop drops it back to false).
    state.cycle_held = false;

    let n = state.nodes.len();
    if n == 0 {
        return Ok(()); // trivially solved
    }
    if n > NETWORK_MAX_NODES {
        return Err(ForwardError::TooManyNodes);
    }

    // Whether a pipe contributes a TOPOLOGY edge for the Kahn ordering. Default:
    // every pipe (state-independent — byte-identical historical behaviour). With
    // cycle-keep: only ON pipes, so toggling a pipe genuinely opens/closes a loop.
    let cycle_keep = state.cycle_keep;
    let edge_active = |p: &Pipe| !cycle_keep || p.is_on();

    // --- Step 1: in-degree from pipe topology. ---
    let mut in_deg = vec![0i32; n];
    for p in &state.pipes {
        if p.to >= n || p.from >= n {
            return Err(ForwardError::BadEdge);
        }
        if edge_active(p) {
            in_deg[p.to] += 1;
        }
    }

    // --- Step 2: seed queue with sources (in_deg == 0 OR is_input). ---
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    let mut enqueued = vec![false; n];
    for i in 0..n {
        if in_deg[i] == 0 || state.nodes[i].is_input {
            queue.push(i);
            enqueued[i] = true;
        }
    }

    // --- Step 3: process queue. ---
    let mut head = 0usize;
    let mut processed = 0usize;
    while head < queue.len() {
        let idx = queue[head];
        head += 1;

        // Compute this node's activation.
        let activation = if state.nodes[idx].is_input {
            i32::from(state.inputs.get(idx).copied().unwrap_or(0) != 0)
        } else {
            let mut sum = 0i32;
            for p in &state.pipes {
                if p.to != idx || !p.is_on() {
                    continue;
                }
                // from-index already bounds-checked in step 1.
                sum += state.nodes[p.from].activation;
            }
            i32::from(sum >= state.nodes[idx].threshold)
        };
        state.nodes[idx].activation = activation;
        processed += 1;

        // Walk outgoing pipes; decrement each successor's in-degree. Must use
        // the SAME edge predicate as Step 1 so in-degree bookkeeping stays
        // consistent (only ON pipes count under cycle-keep).
        for p in &state.pipes {
            if p.from != idx || !edge_active(p) {
                continue;
            }
            let to = p.to;
            in_deg[to] -= 1;
            if in_deg[to] <= 0 && !enqueued[to] {
                queue.push(to);
                enqueued[to] = true;
            }
        }
    }

    // --- Step 4: cycle detection (and the L12 `while(true)` keep). ---
    //
    // OPERATOR NOTE. If Kahn's pass processed fewer nodes than exist, some nodes
    // never reached in-degree zero — they depend on each other in a ring. The
    // thought bites its own tail and never reaches the output. In the story this
    // is denial: a remembering that loops forever and never lets the process
    // `exit`. Truthful: `processed < n` ⇔ a cycle, exactly.
    if processed < n {
        // DEFAULT (cycle_keep == false): the engine refuses the ring as an
        // error — denial is *unsolvable*. `puzzle_interact` rolls the move back,
        // pulling the player out of the loop. This is byte-identical to the
        // historical behaviour: a level that never sets `cycle_keep` takes this
        // branch and sees the same `Err(Cycle)` it always has.
        if !state.cycle_keep {
            return Err(ForwardError::Cycle);
        }

        // L12 `while(true)` — the ring is allowed to CLOSE on purpose. We do not
        // loop forever to "settle" it (that would be non-terminating); instead we
        // define the kept cycle's terminal state directly and deterministically:
        //
        //   * Every node that was already processed by Kahn keeps the activation
        //     the topological pass gave it (the acyclic prefix is unaffected).
        //   * Every node in the cyclic remainder LATCHES ON (`activation = 1`).
        //     "remember(); forget();" — the loop holds its remembered value; the
        //     museum-of-wounds keeps the face lit. Latching to a constant is the
        //     well-defined fixed point of "the loop stands": it is independent of
        //     visit order, so the result is deterministic, and it is a SINGLE
        //     bounded pass over the remainder, so evaluation always terminates.
        //
        // We mark `cycle_held` so `refresh_solved` reads this as the `while(true)`
        // ending rather than a normal door win.
        for (i, &done) in enqueued.iter().enumerate() {
            if !done {
                state.nodes[i].activation = 1;
            }
        }
        state.cycle_held = true;
        return Ok(());
    }
    Ok(())
}

/// What actually happened when the player acted on a cell — the observable
/// result of [`puzzle_interact_ex`].
///
/// The boolean [`puzzle_interact`] collapses every non-[`Toggled`] arm to
/// `false`, which is why the budget refusal needs this richer type: the caller
/// (HUD / audio) must be able to tell "refused: no power" apart from "there was
/// nothing there" and from "that would have closed a loop".
///
/// [`Toggled`]: InteractOutcome::Toggled
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractOutcome {
    /// No pipe switch maps to that cell — the press landed on empty space.
    /// No mutation.
    NoTarget,
    /// The pipe toggled and the network re-evaluated. `now_on` is the pipe's
    /// state AFTER the toggle, so a caller can distinguish powering a wire up
    /// from shutting it down.
    Toggled {
        /// Index of the pipe that toggled.
        pipe: usize,
        /// The pipe's state AFTER the toggle, so a caller can distinguish
        /// powering a wire up from shutting it down.
        now_on: bool,
    },
    /// **Refused — no power.** Switching this pipe `On` would have put more than
    /// [`PuzzleState::power_budget`] pipes on the grid at once. The pipe stays
    /// `Off` and NOTHING else changed: no forward pass, no `solved` refresh, no
    /// `attempts` bump. `used`/`budget` are the readout at the moment of refusal
    /// (`used == budget`, since that is the only way to get here).
    RefusedNoPower {
        /// Index of the pipe whose switch-on was refused.
        pipe: usize,
        /// Pipes drawing power at the moment of refusal.
        used: u32,
        /// The level's power budget (always equal to `used` here).
        budget: u32,
    },
    /// **Refused — the toggle would have closed a graph cycle** on a level that
    /// does not set `cycle_keep`. The toggle is rolled back and the forward pass
    /// re-run from the restored state, so the world is exactly as it was.
    RefusedCycle {
        /// Index of the pipe whose toggle was rolled back.
        pipe: usize,
    },
}

impl InteractOutcome {
    /// Did the world actually change? This is exactly the boolean
    /// [`puzzle_interact`] returns.
    #[inline]
    pub fn changed(self) -> bool {
        matches!(self, InteractOutcome::Toggled { .. })
    }

    /// Did a pipe exist at the interacted cell (whether or not the toggle was
    /// allowed)? Useful for feedback that should acknowledge "you hit a switch"
    /// even when the switch refused.
    #[inline]
    pub fn hit_a_switch(self) -> bool {
        !matches!(self, InteractOutcome::NoTarget)
    }
}

/// Toggle the pipe whose switch cell maps to world `(x, y, z)`, re-run the
/// forward pass, re-check win, and report **what happened**.
///
/// Faithful port of legacy `puzzle_interact`, plus the v0.2 power-budget rule.
/// Validate-before-mutate throughout: every refusal path leaves the world
/// exactly as it was found, never half-applied.
///
/// 1. No pipe at that cell → [`InteractOutcome::NoTarget`], no mutation.
/// 2. **Power budget.** If the pipe is currently `Off` and the level declares a
///    finite [`PuzzleState::power_budget`] that is already fully drawn
///    ([`PuzzleState::power_available`] is false), the switch is REFUSED:
///    [`InteractOutcome::RefusedNoPower`], the pipe stays `Off`, and no forward
///    pass runs. Switching a pipe `Off` is NEVER refused — it frees a unit, so
///    the player can always dig themselves back out. (This is what keeps the
///    constrained state space connected: from any legal configuration you can
///    reach any other legal configuration by switching things off first.)
/// 3. Otherwise toggle `On`↔`Off`, then re-run [`network_forward`]. If the
///    forward pass errors (a cycle on a non-`cycle_keep` level), the toggle is
///    ROLLED BACK and [`InteractOutcome::RefusedCycle`] is returned so the state
///    stays consistent. On success, [`PuzzleState::refresh_solved`] refreshes
///    `solved`, `attempts` is bumped, and [`InteractOutcome::Toggled`] returns.
///
/// The budget check is deliberately BEFORE the toggle rather than after: a
/// refusal must not run a forward pass at all, or a sequence-lock level could
/// silently advance its lock on a move the player was not allowed to make.
pub fn puzzle_interact_ex(state: &mut PuzzleState, x: i32, y: i32, z: i32) -> InteractOutcome {
    let Some(idx) = state.pipe_at_switch(x, y, z) else {
        return InteractOutcome::NoTarget;
    };

    let old = state.pipes[idx].state;

    // --- POWER BUDGET: refuse an ON-switch that would exceed the cap. ---
    if old == PipeState::Off && !state.power_available() {
        return InteractOutcome::RefusedNoPower {
            pipe: idx,
            used: state.power_used(),
            budget: state.power_budget,
        };
    }

    // Toggle.
    state.pipes[idx].state = match old {
        PipeState::Off => PipeState::On,
        PipeState::On => PipeState::Off,
    };

    // Validate-before-commit: roll back on a cycle.
    if network_forward(state).is_err() {
        state.pipes[idx].state = old;
        let _ = network_forward(state);
        return InteractOutcome::RefusedCycle { pipe: idx };
    }

    state.refresh_solved();
    state.attempts += 1;
    InteractOutcome::Toggled {
        pipe: idx,
        now_on: state.pipes[idx].is_on(),
    }
}

/// Toggle the pipe at world `(x, y, z)` and report whether the world CHANGED.
///
/// The historical boolean seam, kept for every existing caller and unchanged in
/// meaning: `true` iff a pipe actually toggled. It is now a thin wrapper over
/// [`puzzle_interact_ex`] — callers that need to distinguish the *reason* a
/// press did nothing (no switch there / no power / would close a loop) should
/// call that instead.
#[inline]
pub fn puzzle_interact(state: &mut PuzzleState, x: i32, y: i32, z: i32) -> bool {
    puzzle_interact_ex(state, x, y, z).changed()
}

/// Set the input value of node `idx` to `value` (`0`/`1`), re-run the forward
/// pass, and refresh `solved`. Returns `true` iff the input actually changed
/// (so the caller can re-solve / re-render only on a real change).
///
/// This is the runtime seam for the **floor-zone input** feature
/// ([`re_level::ZoneInput`]): the game layer computes whether the player stands
/// in a zone and pushes the resulting `0`/`1` here, exactly as
/// [`puzzle_interact`] pushes a pipe toggle. Rules:
///
/// * `idx` out of range, or the node is NOT an input → no-op, returns `false`
///   (an input value only means something on an `is_input` node; setting a
///   hidden/output node's input would be silently overwritten by the forward
///   pass anyway, so we refuse it).
/// * `value` is normalised to `0`/`1` (any non-zero ⇒ `1`).
/// * No change (the input already holds that value) → returns `false`, no
///   forward pass.
/// * A change re-runs [`network_forward`] + [`check_win`]. A forward-pass error
///   (a cycle — impossible for a v0.1 DAG, guarded) ROLLS BACK the input so the
///   state stays consistent, and returns `false`.
pub fn set_input_value(state: &mut PuzzleState, idx: usize, value: i32) -> bool {
    let Some(node) = state.nodes.get(idx) else {
        return false;
    };
    if !node.is_input {
        return false;
    }
    let v = i32::from(value != 0);
    let old = state.inputs.get(idx).copied().unwrap_or(0);
    if old == v {
        return false;
    }
    state.inputs[idx] = v;
    if network_forward(state).is_err() {
        state.inputs[idx] = old;
        let _ = network_forward(state);
        return false;
    }
    state.refresh_solved();
    true
}

/// Win check: `true` iff every node in `door_targets` has `activation == 1`.
///
/// Port of legacy `puzzle_check_win`. An empty `door_targets` list returns
/// `false` — a level with no win condition cannot be won by accident. An
/// out-of-range target index also fails the check (defensive).
pub fn check_win(state: &PuzzleState, door_targets: &[usize]) -> bool {
    if door_targets.is_empty() {
        return false;
    }
    door_targets.iter().all(|&idx| {
        state
            .nodes
            .get(idx)
            .is_some_and(|node| node.activation == 1)
    })
}

/// Combined win check for the L12 `exit 0` **inverted** mode: every
/// `door_targets` node must FIRE *and* every `target_off` node must NOT fire.
///
/// This is the win seam [`PuzzleState::refresh_solved`] uses in the ordinary
/// (non-sequence, non-cycle) path. It is a strict superset of [`check_win`]:
///
/// * With an EMPTY `target_off` (the default for every level that does not
///   declare `[door] target_off`) it returns EXACTLY
///   `check_win(state, door_targets)` — byte-identical, so default behaviour and
///   every existing test are unchanged.
/// * With a non-empty `target_off` the player wins only when the inverted
///   targets are dark: `REC` must read 0 (the player pruned the self-feed) while
///   the ordinary `door_targets` (e.g. `EXIT`) fire.
///
/// An out-of-range `target_off` index is treated as NOT satisfying the "must be
/// off" requirement (defensive: a malformed inverted target can never silently
/// grant a win). A `target_off`-only level (empty `door_targets`) still returns
/// `false`, because a level with no positive win condition cannot be won —
/// [`check_win`]'s empty-targets rule is preserved.
pub fn check_win_inverted(
    state: &PuzzleState,
    door_targets: &[usize],
    target_off: &[usize],
) -> bool {
    // The positive half is exactly the historical check (incl. the empty rule).
    if !check_win(state, door_targets) {
        return false;
    }
    // The inverted half: every named node must be present AND dark.
    target_off.iter().all(|&idx| {
        state
            .nodes
            .get(idx)
            .is_some_and(|node| node.activation == 0)
    })
}

// ===========================================================================
// Brute-force solvability analysis (static, non-mutating)
// ===========================================================================

/// The hard cap on the number of pipes a [`solve_brute_force`] sweep will
/// enumerate exhaustively. With one bit per pipe the search is `2^n`, so this
/// keeps the worst case at `2^20 ≈ 1M` forward passes — fast for v0.1 graphs.
/// A network with more pipes than this is reported [`Solvability::TooManyPipes`]
/// rather than searched.
pub const SOLVE_MAX_PIPES: usize = 20;

/// The verdict of a static [`solve_brute_force`] sweep over every pipe
/// on/off combination.
///
/// The pipes that participate in the search are the **switchable** pipes — the
/// ones with a `switch_cell` (a player can actually toggle them). Pipes with no
/// switch are held at their authored state for the whole sweep (the player can
/// never change them), so a level is only "solvable" if it can be won by
/// toggling the switchable pipes alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Solvability {
    /// The level is already won at its authored initial configuration — no
    /// toggling required. A WARN: there is no puzzle to perform.
    TriviallySolvedAtLoad,
    /// A winning configuration exists. `min_toggles` is the fewest switch
    /// flips (from the authored initial state) that reach a win; `winning_mask`
    /// is the desired ON-set over the *switchable* pipes (bit `i` ↔ the `i`-th
    /// switchable pipe) for one such minimal solution.
    Solvable {
        /// Fewest switch flips (from the authored initial state) that reach a win.
        min_toggles: u32,
        /// The desired ON-set over the *switchable* pipes (bit `i` ↔ the
        /// `i`-th switchable pipe) for one such minimal solution.
        winning_mask: u32,
        /// How many pipes were switchable (searched).
        switchable: usize,
    },
    /// No combination of switch states wins — the level cannot be solved by the
    /// player. A FAIL.
    Unsolvable,
    /// There are too many switchable pipes to brute-force (`> SOLVE_MAX_PIPES`).
    TooManyPipes {
        /// The count that would have to be searched.
        switchable: usize,
    },
}

/// Statically determine whether `state` is solvable by toggling only its
/// **switchable** pipes, reusing the real [`network_forward`] + [`check_win`]
/// passes — it never reimplements the network math.
///
/// The sweep enumerates every on/off assignment of the switchable pipes (held
/// non-switchable pipes at their authored state), runs the forward pass, and
/// records every assignment that wins. Among the winners it picks the one whose
/// Hamming distance from the authored initial switch states is smallest — that
/// distance is the **minimum number of toggles** a player must perform.
///
/// The passed `state` is restored to its original pipe states + forward pass
/// before returning, so this is a pure analysis with no observable mutation.
pub fn solve_brute_force(state: &mut PuzzleState) -> Solvability {
    // Indices of the switchable pipes (the only ones the player can toggle).
    let switchable: Vec<usize> = state
        .pipes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.switch_cell.is_some())
        .map(|(i, _)| i)
        .collect();
    let k = switchable.len();

    // No win condition at all ⇒ unwinnable by construction. This is the
    // historical `door_targets.is_empty()` bail, widened only by the additive
    // L12 modes: a `cycle_keep` level can win by closing a loop even with no
    // door targets. A level with neither door targets NOR cycle-keep can never
    // win (byte-identical to before for every existing level, all of which have
    // door targets and `cycle_keep == false`).
    if state.door_targets.is_empty() && !state.cycle_keep {
        return Solvability::Unsolvable;
    }

    // Snapshot to restore on every exit path.
    let saved: Vec<PipeState> = state.pipe_states();
    let targets = state.door_targets.clone();
    let off = state.target_off.clone();

    // Already won at the authored configuration? Uses the SAME full win
    // semantics the runtime does (`solve_state_won`): the inverted-target +
    // cycle-keep modes are respected so lvlcheck's verdict matches play. With the
    // defaults (empty `target_off`, `cycle_keep == false`) this is exactly the
    // old `check_win(state, &targets)`.
    let trivially_solved = {
        let _ = network_forward(state);
        solve_state_won(state, &targets, &off)
    };

    if k > SOLVE_MAX_PIPES {
        // Restore and bail — too large to enumerate.
        restore_pipes(state, &saved);
        return if trivially_solved {
            Solvability::TriviallySolvedAtLoad
        } else {
            Solvability::TooManyPipes { switchable: k }
        };
    }

    // Authored initial ON-set over the switchable pipes (bit i ↔ switchable[i]).
    let initial_mask: u32 = (0..k)
        .filter(|&i| saved[switchable[i]] == PipeState::On)
        .fold(0u32, |m, i| m | (1u32 << i));

    // POWER BUDGET: pipes the player cannot switch still DRAW power (see
    // `PuzzleState::power_used`), so the fixed part of the draw is counted once
    // here and added to each candidate assignment's popcount. `0` ⇒ unlimited.
    let budget = state.power_budget;
    let fixed_on: u32 = (0..saved.len())
        .filter(|i| !switchable.contains(i) && saved[*i] == PipeState::On)
        .count() as u32;

    let total: u32 = 1u32 << k;
    let mut best: Option<(u32, u32)> = None; // (toggles, mask)

    for s in 0..total {
        // Skip any assignment the level's power budget forbids: it is not a
        // configuration the player could ever be standing in, so reporting it as
        // the solution would make lvlcheck lie about solvability.
        if budget > 0 && fixed_on + s.count_ones() > budget {
            continue;
        }
        // Apply this assignment to the switchable pipes.
        for (i, &pi) in switchable.iter().enumerate() {
            state.pipes[pi].state = if (s >> i) & 1 == 1 {
                PipeState::On
            } else {
                PipeState::Off
            };
        }
        if network_forward(state).is_err() {
            // Only a `cycle_keep == false` cycle (or a malformed graph) errors;
            // with cycle-keep the loop-closing assignment returns Ok + a held
            // cycle, which `solve_state_won` reads as a `while(true)` win.
            continue;
        }
        if solve_state_won(state, &targets, &off) {
            let toggles = (s ^ initial_mask).count_ones();
            match best {
                Some((bt, _)) if bt <= toggles => {}
                _ => best = Some((toggles, s)),
            }
        }
    }

    restore_pipes(state, &saved);

    match best {
        None => Solvability::Unsolvable,
        Some((0, _)) => Solvability::TriviallySolvedAtLoad,
        Some((min_toggles, winning_mask)) => Solvability::Solvable {
            min_toggles,
            winning_mask,
            switchable: k,
        },
    }
}

/// Is `state` (already forward-passed) in a WON terminal configuration, under
/// the full L12 win semantics? Mirrors [`PuzzleState::refresh_solved`]'s
/// non-sequence path so the static solver agrees with the runtime:
///
/// * a KEPT cycle that stood (`cycle_held`) is the `while(true)` win, else
/// * the combined inverted check ([`check_win_inverted`]).
///
/// With the defaults (no kept cycle, empty `target_off`) this reduces to
/// `check_win(state, door_targets)` — so the solver is byte-identical for every
/// pre-L12 level. The sequence-lock is intentionally NOT consulted here: it is
/// an ordered, stateful machine that the static single-shot sweep cannot drive,
/// and lvlcheck already treats sequence levels via their own path.
fn solve_state_won(state: &PuzzleState, door_targets: &[usize], target_off: &[usize]) -> bool {
    if state.cycle_held {
        return true;
    }
    check_win_inverted(state, door_targets, target_off)
}

/// Restore pipe states from a snapshot and re-run the forward pass (helper for
/// [`solve_brute_force`]'s exit paths).
fn restore_pipes(state: &mut PuzzleState, snapshot: &[PipeState]) {
    for (pipe, &s) in state.pipes.iter_mut().zip(snapshot) {
        pipe.state = s;
    }
    let _ = network_forward(state);
}

// ===========================================================================
// Power-budget feasibility (stateful reachability, not a stateless sweep)
// ===========================================================================

/// Whether a level can still be WON under a given power budget.
///
/// Deliberately three-valued: an un-searchable network must not be silently
/// reported as unwinnable, because the whole point of the safety test is that a
/// budget which makes a level unwinnable is a catastrophic regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetFeasibility {
    /// A winning state is REACHABLE from the authored start under the budget.
    Winnable,
    /// The whole reachable state space was searched and none of it wins.
    Unwinnable,
    /// More than [`SOLVE_MAX_PIPES`] switchable pipes — not searched.
    TooManyPipes {
        /// The count that would have to be searched.
        switchable: usize,
    },
}

/// Can the player reach a WON state from this level's authored start while never
/// exceeding `budget` simultaneously-ON pipes? (`budget == 0` ⇒ unlimited.)
///
/// **Why this is not `solve_brute_force`.** That function is a STATELESS sweep:
/// it enumerates pipe assignments, forward-passes each in isolation, and asks
/// `check_win`. That is wrong for any level whose win condition is a state
/// machine. On `L11-inner-circuit` the [`SequenceLock`] advances as a side effect
/// of the sweep's own enumeration, so the sweep happily reports a 1-toggle
/// solution for a lock that genuinely needs the output driven through three
/// distinct patterns in order. A budget derived from that number would be a lie.
///
/// So this is a real breadth-first search over `(pipe_mask, sequence_progress)`,
/// where every edge is an actual [`puzzle_interact_ex`] call at the pipe's
/// authored switch cell. It never reimplements a rule: refusals (no power, cycle
/// rollback), the lock's monotone advance, the inverted-target and cycle-keep
/// win modes are all whatever the runtime says they are.
///
/// Non-switchable pipes are held at their authored state (the player can't touch
/// them) but still count toward the draw, exactly as [`PuzzleState::power_used`]
/// counts them.
///
/// `state` is only read; the search runs on clones.
pub fn win_reachable_under_budget(state: &PuzzleState, budget: u32) -> BudgetFeasibility {
    let switchable: Vec<usize> = state
        .pipes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.switch_cell.is_some())
        .map(|(i, _)| i)
        .collect();
    let k = switchable.len();
    if k > SOLVE_MAX_PIPES {
        return BudgetFeasibility::TooManyPipes { switchable: k };
    }

    // Switch coordinates, resolved once. A switchable pipe whose `switch_cell`
    // index is off the end of its path has no reachable switch: treat it as
    // unswitchable rather than panicking (a level-authoring defect that
    // `campaign_walkable` reports far more usefully than a solver crash).
    let coords: Vec<Option<Coord3>> = switchable
        .iter()
        .map(|&pi| state.pipes[pi].switch_coord())
        .collect();

    let steps = state.sequence.patterns.len() + 1; // progress ∈ 0..=patterns.len()
    let mask_of = |st: &PuzzleState| -> u32 {
        switchable
            .iter()
            .enumerate()
            .filter(|(_, &pi)| st.pipes[pi].is_on())
            .fold(0u32, |m, (bit, _)| m | (1u32 << bit))
    };

    // Node key → a dense visited grid over (mask, progress).
    let mut seen = vec![false; (1usize << k) * steps];
    let key = |mask: u32, prog: usize| (mask as usize) * steps + prog;

    // The authored start, evaluated exactly as the runtime evaluates it at load.
    let mut start = state.clone();
    start.power_budget = budget;
    start.sequence.reset_progress();
    if network_forward(&mut start).is_err() {
        // A start configuration that does not even forward-pass cannot be played.
        return BudgetFeasibility::Unwinnable;
    }
    start.refresh_solved();
    if start.solved {
        return BudgetFeasibility::Winnable;
    }

    let start_key = (mask_of(&start), start.sequence.progress);
    seen[key(start_key.0, start_key.1)] = true;
    let mut frontier: Vec<(u32, usize)> = vec![start_key];

    while let Some((mask, prog)) = frontier.pop() {
        for coord in &coords {
            let Some((sx, sy, sz)) = *coord else {
                continue;
            };
            // Rebuild the node's world, then take the real interact edge.
            let mut st = state.clone();
            st.power_budget = budget;
            for (b, &qi) in switchable.iter().enumerate() {
                st.pipes[qi].state = if (mask >> b) & 1 == 1 {
                    PipeState::On
                } else {
                    PipeState::Off
                };
            }
            st.sequence.progress = prog;
            if network_forward(&mut st).is_err() {
                continue;
            }
            if !puzzle_interact_ex(&mut st, sx, sy, sz).changed() {
                continue; // refused (no power / cycle) — not a reachable edge
            }
            if st.solved {
                return BudgetFeasibility::Winnable;
            }
            let next = (mask_of(&st), st.sequence.progress);
            let idx = key(next.0, next.1);
            if !seen[idx] {
                seen[idx] = true;
                frontier.push(next);
            }
        }
    }
    BudgetFeasibility::Unwinnable
}

/// The SMALLEST power budget under which this level is still winnable, or `None`
/// if no budget makes it winnable (or it has too many switchable pipes to
/// search).
///
/// This is the number `[room] power_budget` should be set to for a level that
/// wants to be as tight as it can be: one lower and the level is unwinnable.
/// Computed by running [`win_reachable_under_budget`] upward from `1`; the
/// upper bound is the pipe count (at which the budget cannot bind at all).
///
/// Note this is a MAXIMUM-CONCURRENCY figure, not a toggle count: a level may
/// need many flips over time while never having more than a couple of wires hot.
pub fn min_power_budget(state: &PuzzleState) -> Option<u32> {
    let n = state.pipes.len() as u32;
    for budget in 1..=n {
        match win_reachable_under_budget(state, budget) {
            BudgetFeasibility::Winnable => return Some(budget),
            BudgetFeasibility::Unwinnable => {}
            BudgetFeasibility::TooManyPipes { .. } => return None,
        }
    }
    None
}

// ===========================================================================
// Bevy plugin layer
// ===========================================================================

/// A request to interact with the world cell `(x, y, z)`.
///
/// The input layer emits this (as a Bevy [`Message`]) when the player presses
/// the interact action while looking at a cell; [`PuzzlePlugin`]'s system
/// drains it into [`puzzle_interact`]. Keeping it a message means the puzzle
/// crate never reaches into the input crate.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractAt {
    /// Cell X coordinate.
    pub x: i32,
    /// Cell Y coordinate.
    pub y: i32,
    /// Cell Z coordinate (storey).
    pub z: i32,
}

impl InteractAt {
    /// Construct from an `(x, y, z)` cell.
    #[inline]
    pub fn cell(x: i32, y: i32, z: i32) -> Self {
        InteractAt { x, y, z }
    }
}

/// The RESULT of an applied [`InteractAt`], published back out as a Bevy
/// [`Message`] so the presentation layers can react to what actually happened.
///
/// This closes a real defect: the host game used to PREDICT the interact ack sound
/// from "is there a switch under the crosshair?", which meant a REJECTED
/// interaction sounded exactly like an accepted one. With a power budget in play
/// that is unacceptable — a refusal that sounds like a success reads as a broken
/// game. The outcome now travels with the event, and the sound is chosen from
/// it. `re_puzzle` still knows nothing about audio or the HUD; it only reports.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractResolved {
    /// The cell the player acted on (echoed back so a listener can locate it).
    pub at: InteractAt,
    /// What the puzzle runtime did about it.
    pub outcome: InteractOutcome,
}

/// The solved flag, surfaced as a resource so other systems (HUD, level
/// advance, save) can read it without touching [`PuzzleState`].
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Solved(pub bool);

/// The Bevy plugin wiring the puzzle network into the app.
///
/// Inserts a default [`PuzzleState`] + [`Solved`] resource, registers the
/// [`InteractAt`] message, and runs [`apply_interactions`] in `Update`. Build
/// the network into the world with [`spawn_from_level`] (or insert a
/// [`PuzzleState::from_level`] result directly) at level-load time.
#[derive(Default)]
pub struct PuzzlePlugin;

impl Plugin for PuzzlePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<InteractAt>()
            .add_message::<InteractResolved>()
            .init_resource::<PuzzleState>()
            .init_resource::<Solved>()
            .add_systems(Update, apply_interactions);
    }
}

/// `Update` system: drain [`InteractAt`] messages into [`puzzle_interact_ex`],
/// publish each result as an [`InteractResolved`] message, and mirror the
/// resulting solved state into the [`Solved`] resource.
pub fn apply_interactions(
    mut messages: MessageReader<InteractAt>,
    mut state: ResMut<PuzzleState>,
    mut solved: ResMut<Solved>,
    mut resolved: MessageWriter<InteractResolved>,
) {
    for m in messages.read() {
        let outcome = puzzle_interact_ex(&mut state, m.x, m.y, m.z);
        resolved.write(InteractResolved { at: *m, outcome });
    }
    if solved.0 != state.solved {
        solved.0 = state.solved;
    }
}

/// Build the puzzle network from a [`Level`] and insert it (plus the matching
/// [`Solved`] resource) into the ECS world via `commands`.
///
/// The ECS counterpart of [`PuzzleState::from_level`]. On a malformed / cyclic
/// network the error is returned and nothing is inserted (the caller decides
/// whether a bad level is fatal). Pipes/nodes live in the [`PuzzleState`]
/// resource rather than as individual entities — the network is small (v0.1
/// caps at a handful of nodes) and the forward pass wants them contiguous.
pub fn spawn_from_level(commands: &mut Commands, level: &Level) -> Result<(), ForwardError> {
    let state = PuzzleState::from_level(level)?;
    let solved = Solved(state.solved);
    commands.insert_resource(state);
    commands.insert_resource(solved);
    Ok(())
}

#[cfg(test)]
mod power_tests;
#[cfg(test)]
mod tests;
