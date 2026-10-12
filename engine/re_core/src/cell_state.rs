//! `cell_state` — the **state bus**: a generic per-cell-tag state map.
//!
//! [`CellStateMap`] lets the renderer SEE a per-cell live state without knowing
//! what that state MEANS. It maps a cell's game `tag` (`u16` — the per-cell
//! `side_data` / `material` key the world already carries) to two small
//! generic values:
//!
//! * `state: u32` — a packed, opaque state word (the bridge in `re_game` encodes
//!   puzzle semantics — powered / dead / decoy / won, pipe on/off — into it; the
//!   renderer only indexes a glyph family by it).
//! * `pulse: f32` — a transient `[0,1]` emissive/anim KICK (a tag that just fired,
//!   a switch that just toggled, a cycle rejection). Decays to 0 (handled by the
//!   bridge each frame); the renderer reads it as a small additive emissive term.
//!
//! **NO puzzle semantics live here.** This type is pure data shared by `re_render`
//! (reads `u32`/`f32` per tag — never depends on `re_puzzle`) and the `re_game`
//! bridge (writes it from `PuzzleState`). The split mirrors the engine contract's
//! `EmitterSpec`/`Emitter` discipline: the generic carrier is engine-side, the
//! semantics are game-side.
//!
//! ## Default = today's frame
//!
//! A default [`CellStateMap`] is EMPTY: `state(tag) == 0` and `pulse(tag) == 0` for
//! every tag. A renderer fed an empty (or absent) map shades byte-for-byte as it
//! does with no state bus at all — so this layer no-ops until the bridge populates
//! it. Golden frames hold.
//!
//! ## Storage
//!
//! Dense `Vec` indexed by tag (spec open-Q1 REC): puzzle node tags are small,
//! contiguous node indices (`0..n_nodes`), so a dense vector is the fast, simple
//! choice — a tag beyond the populated range reads the EMPTY default (`0`). The
//! vectors grow on demand as tags are written; an unused map allocates nothing.

use bevy::prelude::Resource;

/// A generic per-tag state bus: `tag(u16) → (state: u32, pulse: f32)`.
///
/// Empty by default (every tag reads `state 0` / `pulse 0` → today's look). The
/// `re_game` bridge fills it from the live `PuzzleState`; `re_render` reads it
/// (only `u32`/`f32` per tag) when shading a cell whose `material` tag matches.
///
/// A Bevy [`Resource`] so the bridge can `ResMut` it and the render system can read
/// it — but every method works standalone (no `World`), so unit tests build one
/// directly.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct CellStateMap {
    /// Dense state word per tag; tags beyond `len` read the EMPTY default `0`.
    tag_state: Vec<u32>,
    /// Dense pulse per tag; tags beyond `len` read the EMPTY default `0.0`.
    tag_pulse: Vec<f32>,
}

impl CellStateMap {
    /// An empty map — `state(tag) == 0`, `pulse(tag) == 0` for every tag. Allocates
    /// nothing; the renderer shades exactly as it does with no state bus.
    pub const fn new() -> Self {
        CellStateMap {
            tag_state: Vec::new(),
            tag_pulse: Vec::new(),
        }
    }

    /// True iff no tag has been given a non-default state OR pulse — i.e. shading
    /// against this map is byte-identical to shading with no map. (A map that was
    /// written then cleared back to all-zero still reports its allocation here, but
    /// reads default for every tag, so the OUTPUT is unchanged; this is a cheap
    /// `len == 0` check, the common no-op fast path.)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tag_state.is_empty() && self.tag_pulse.is_empty()
    }

    /// The opaque state word for `tag`. EMPTY default `0` for any tag never written
    /// (including every tag when the map is empty). Pure, allocation-free.
    #[inline]
    pub fn state(&self, tag: u16) -> u32 {
        self.tag_state.get(tag as usize).copied().unwrap_or(0)
    }

    /// The transient pulse `[0,1]` for `tag`. EMPTY default `0.0` for any tag never
    /// written. Pure, allocation-free.
    #[inline]
    pub fn pulse(&self, tag: u16) -> f32 {
        self.tag_pulse.get(tag as usize).copied().unwrap_or(0.0)
    }

    /// Set the state word for `tag`, growing the dense store as needed (intervening
    /// tags default to `0`). Idempotent; cheap for the small, contiguous node-index
    /// tag space the bridge writes.
    #[inline]
    pub fn set_state(&mut self, tag: u16, state: u32) {
        let i = tag as usize;
        if i >= self.tag_state.len() {
            self.tag_state.resize(i + 1, 0);
        }
        self.tag_state[i] = state;
    }

    /// Set the pulse `[0,1]` for `tag`, growing the dense store as needed.
    #[inline]
    pub fn set_pulse(&mut self, tag: u16, pulse: f32) {
        let i = tag as usize;
        if i >= self.tag_pulse.len() {
            self.tag_pulse.resize(i + 1, 0.0);
        }
        self.tag_pulse[i] = pulse;
    }

    /// Reset every tag to the EMPTY default (`state 0` / `pulse 0`) WITHOUT freeing
    /// the backing storage, so a per-frame rebuild reuses the allocation. After this
    /// the map reads default for every tag (today's look) until re-populated.
    #[inline]
    pub fn clear(&mut self) {
        self.tag_state.fill(0);
        self.tag_pulse.fill(0.0);
    }
}

// ───────────────────────── pipe-cell tag + flow-state contract ──────────────
//
// SHARED CONTRACT (reactive-materials spec §3, R2). A puzzle pipe is stamped into
// the world as a run of `WallTrace` (circuit-trace) GEOMETRY cells. A geometry
// cell's `side_data` is normally a 1-based COLOUR-material id (R1), so a pipe cell
// needs a DISTINCT namespace in the same `u16` field so the renderer can tell "this
// trace cell is a live pipe, read the state bus by its tag" from "this trace cell
// is a coloured surface, read the colour palette". We reserve the HIGH BIT:
//
//   side_data with bit 15 SET  ⇒ a PIPE cell. Low 15 bits = the state-bus pipe tag.
//   side_data with bit 15 CLEAR ⇒ unchanged (0 = none, 1.. = colour id) — today.
//
// Back-compat: every shipped level sets only small colour ids (< 0x8000) or 0, so
// no existing cell carries the flag → geometry/colour dispatch is byte-identical
// and goldens hold. The `re_level` stamper sets the flag; `re_render` routes a
// flagged WallTrace cell to the state bus (keyed by the low bits) instead of the
// colour palette; the `re_game` bridge writes that tag's flow state.

/// The high bit of a geometry cell's `side_data` that marks it as a PUZZLE PIPE
/// cell (a live `WallTrace` carrying a state-bus tag, not a colour id). See the
/// module-level contract. Equal to `1 << 15`.
pub const PIPE_TAG_FLAG: u16 = 0x8000;

/// Mask selecting the pipe-tag payload (the low 15 bits) out of a flagged
/// `side_data`.
pub const PIPE_TAG_MASK: u16 = 0x7FFF;

/// First value of the RESERVED SENTINEL band of the non-pipe `side_data` half
/// (`RESERVED_TAG_BASE ..= PIPE_TAG_MASK`, i.e. `0x7F00..=0x7FFF`).
///
/// The `u16` `side_data` field carries three disjoint namespaces:
///
/// * `0` — no tag; small integers — a node index on an `Object` cell, or a 1-based
///   COLOUR-palette id (R1) on geometry. Both grow from 1 and stay far below this band.
/// * `RESERVED_TAG_BASE ..= PIPE_TAG_MASK` — derived renderer sentinels that must
///   never be read as a node, a colour or a pipe (see [`INERT_OBJECT_TAG`]).
/// * `PIPE_TAG_FLAG ..` — a live puzzle pipe (bit 15 set).
///
/// A sentinel therefore sits BELOW the pipe flag (so [`decode_pipe_tag`] never
/// decodes it) and ABOVE every palette id a level can realistically author.
pub const RESERVED_TAG_BASE: u16 = 0x7F00;

/// `side_data` of an `Object` cell that no puzzle entity backs (an inert `O`,
/// set-dressing F2). It lives in the reserved sentinel band — NOT in the pipe
/// namespace (the old `0xD003` had bit 15 set, so an inert bench decoded as a pipe
/// trace) and not in the node / colour-id range. Derived on every load, never
/// persisted. `re_level::INERT_OBJECT_TAG` and `re_render::INERT_OBJECT_TAG` are
/// re-exports of this one value.
pub const INERT_OBJECT_TAG: u16 = RESERVED_TAG_BASE | 0x03;

/// `side_data` a level may stamp on the DOOR `Object` cell so the renderer shades it
/// as the exit rather than a node (`re_render::DOOR_TAG_SENTINEL` re-exports it).
/// Like [`INERT_OBJECT_TAG`] it was moved out of the pipe namespace: the old
/// `0xD002` had bit 15 set, so [`decode_pipe_tag`] read a marked door as pipe `0x5002`.
pub const DOOR_OBJECT_TAG: u16 = RESERVED_TAG_BASE | 0x02;

/// Every derived sentinel in the reserved band, for the disjointness tests. A new
/// sentinel is added here so the tests cover it.
pub const RESERVED_SENTINELS: [u16; 2] = [DOOR_OBJECT_TAG, INERT_OBJECT_TAG];

const _: () = assert!(DOOR_OBJECT_TAG & PIPE_TAG_FLAG == 0);
const _: () = assert!(DOOR_OBJECT_TAG >= RESERVED_TAG_BASE && DOOR_OBJECT_TAG <= PIPE_TAG_MASK);
const _: () = assert!(DOOR_OBJECT_TAG != INERT_OBJECT_TAG);

// ── Hologram style (`side_data` of a `CellKind::Hologram` cell) ────────────────
//
// A hologram cell's `side_data` is its STYLE. It shares nothing with the pipe /
// sentinel namespaces above (those are read only on trace / object cells), but it
// has its own layout so a `[holo] style` and a `holo_display` colour can tint it:
//
//   0                      — the default hologram blue (the look before styles).
//   1 ..= HOLO_PRESETS.len() — a named preset hue (`HOLO_PRESETS`, 1-based).
//   HOLO_STYLE_RGB | 0xRGB — an authored colour, 4 bits per channel.
//   anything else          — reserved; renders as the default blue.

/// Named hologram hues, in style order (`style = 1` is the first). Each entry is
/// `(name, front colour)`; the back/side contour is derived from the front.
pub const HOLO_PRESETS: [(&str, u32); 8] = [
    ("cyan", 0x0030_E0FF),
    ("green", 0x0040_FF90),
    ("amber", 0x00FF_B040),
    ("red", 0x00FF_4848),
    ("magenta", 0x00FF_48E0),
    ("white", 0x00E8_F0FF),
    ("violet", 0x0098_68FF),
    ("teal", 0x0030_D0B0),
];

/// Flag marking a hologram style as an authored 12-bit colour (`0x1RGB`).
pub const HOLO_STYLE_RGB: u16 = 0x1000;

/// Encode an authored `0xRRGGBB` colour as a hologram style (nearest 4-bit
/// channel each, so `#FF40C0` → `0x1F4B`).
pub fn encode_holo_rgb(rgb: u32) -> u16 {
    let q = |sh: u32| ((((rgb >> sh) & 0xFF) + 8) / 17).min(15) as u16;
    HOLO_STYLE_RGB | (q(16) << 8) | (q(8) << 4) | q(0)
}

/// The style for a preset name (`"magenta"` → 5), case-insensitive.
pub fn holo_style_named(name: &str) -> Option<u16> {
    HOLO_PRESETS
        .iter()
        .position(|(n, _)| n.eq_ignore_ascii_case(name.trim()))
        .map(|i| i as u16 + 1)
}

/// The FRONT colour a hologram style asks for, or `None` for the default blue
/// (style 0, or a reserved value).
pub fn decode_holo_style(style: u16) -> Option<u32> {
    match style {
        0 => None,
        s if (s as usize) <= HOLO_PRESETS.len() => Some(HOLO_PRESETS[s as usize - 1].1),
        s if s & 0xF000 == HOLO_STYLE_RGB => {
            let e = |sh: u16| u32::from((s >> sh) & 0xF) * 17;
            Some((e(8) << 16) | (e(4) << 8) | e(0))
        }
        _ => None,
    }
}

// Compile-time namespace guard: the inert sentinel is inside the reserved band and
// outside the pipe namespace.
const _: () = assert!(INERT_OBJECT_TAG & PIPE_TAG_FLAG == 0);
const _: () = assert!(INERT_OBJECT_TAG >= RESERVED_TAG_BASE && INERT_OBJECT_TAG <= PIPE_TAG_MASK);

/// Encode a pipe's state-bus `tag` (its low-15-bit id) into a `side_data` value
/// with the [`PIPE_TAG_FLAG`] set. The `tag` is masked to 15 bits (the v0.1 puzzle
/// has far fewer than 32 768 pipes, so this never clips a real id). Pure.
#[inline]
pub const fn encode_pipe_side_data(tag: u16) -> u16 {
    PIPE_TAG_FLAG | (tag & PIPE_TAG_MASK)
}

/// If `side_data` is a flagged pipe cell, return its state-bus tag (the low 15
/// bits); otherwise `None` (a colour id / plain geometry). Pure, allocation-free.
#[inline]
pub const fn decode_pipe_tag(side_data: u16) -> Option<u16> {
    if side_data & PIPE_TAG_FLAG != 0 {
        Some(side_data & PIPE_TAG_MASK)
    } else {
        None
    }
}

/// Flow direction along a pipe's authored path, packed into the pipe cell's state
/// word. The renderer reads it to scroll the flow pattern toward the OUTPUT end
/// (input→output) when the pipe is powered. `Forward` scrolls along increasing
/// path index (the authored input→output order); `Backward` scrolls the other way
/// (for a pipe whose authored path runs output→input). `None` is the resting value
/// an empty bus yields → a static, cold trace (today's look).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowDir {
    /// Unpowered / no flow — static, cold trace. The state-word `0` value (so an
    /// EMPTY bus reads this for every tag → byte-identical to today).
    None,
    /// Powered, scrolling toward increasing path index (authored input→output).
    Forward,
    /// Powered, scrolling toward decreasing path index.
    Backward,
}

// Pipe-cell STATE WORD layout (the `u32` the bus carries for a pipe tag):
//   bit 0      : powered (1) / unpowered (0)
//   bits 1..=2 : flow direction (0 = none, 1 = forward, 2 = backward)
// `state == 0` ⇒ unpowered + no flow ⇒ the resting cold trace (empty-bus default).
const PIPE_POWERED_BIT: u32 = 1;
const PIPE_DIR_SHIFT: u32 = 1;
const PIPE_DIR_MASK: u32 = 0b11;

/// Encode a pipe cell's flow [`FlowDir`] into the state-bus `u32` word. An
/// unpowered pipe ([`FlowDir::None`]) encodes to `0` (the empty-bus default → a
/// cold, static trace), so a level whose bus never powers a pipe is byte-identical
/// to today. Pure.
#[inline]
pub const fn encode_flow_state(dir: FlowDir) -> u32 {
    match dir {
        FlowDir::None => 0,
        FlowDir::Forward => PIPE_POWERED_BIT | (1 << PIPE_DIR_SHIFT),
        FlowDir::Backward => PIPE_POWERED_BIT | (2 << PIPE_DIR_SHIFT),
    }
}

/// Decode a pipe cell's flow [`FlowDir`] from the state-bus `u32` word (the inverse
/// of [`encode_flow_state`]). An unpowered word (`0`, or any word with the powered
/// bit clear) decodes to [`FlowDir::None`]. Pure — the renderer reads this to pick
/// the flow scroll, never knowing it came from a puzzle.
#[inline]
pub const fn decode_flow_state(state: u32) -> FlowDir {
    if state & PIPE_POWERED_BIT == 0 {
        return FlowDir::None;
    }
    match (state >> PIPE_DIR_SHIFT) & PIPE_DIR_MASK {
        1 => FlowDir::Forward,
        2 => FlowDir::Backward,
        // A powered word with no/invalid direction bucket still reads as flowing
        // forward (the default along the authored path) rather than going cold.
        _ => FlowDir::Forward,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_map_reads_default_everywhere() {
        let m = CellStateMap::new();
        assert!(m.is_empty());
        for tag in [0u16, 1, 7, 255, 4096, u16::MAX] {
            assert_eq!(m.state(tag), 0, "empty state default");
            assert_eq!(m.pulse(tag), 0.0, "empty pulse default");
        }
    }

    #[test]
    fn set_then_read_round_trips() {
        let mut m = CellStateMap::new();
        m.set_state(3, 0xABCD);
        m.set_pulse(3, 0.5);
        assert_eq!(m.state(3), 0xABCD);
        assert_eq!(m.pulse(3), 0.5);
        // Untouched neighbours still read the default.
        assert_eq!(m.state(2), 0);
        assert_eq!(m.state(4), 0);
        assert_eq!(m.pulse(0), 0.0);
        assert!(!m.is_empty());
    }

    #[test]
    fn growth_fills_intervening_tags_with_default() {
        let mut m = CellStateMap::new();
        m.set_state(5, 9);
        for tag in 0..5u16 {
            assert_eq!(m.state(tag), 0, "intervening tag {tag} defaults to 0");
        }
        assert_eq!(m.state(5), 9);
        // A tag beyond what was written still reads default (no OOB panic).
        assert_eq!(m.state(6), 0);
    }

    #[test]
    fn clear_resets_to_default_keeping_capacity() {
        let mut m = CellStateMap::new();
        m.set_state(2, 7);
        m.set_pulse(2, 1.0);
        m.clear();
        assert_eq!(m.state(2), 0);
        assert_eq!(m.pulse(2), 0.0);
        // `clear` keeps the storage, so `is_empty` (a len check) is still false —
        // but every READ is the default, which is what byte-identity needs.
        assert!(!m.is_empty());
    }

    #[test]
    fn pipe_side_data_round_trips_and_is_disjoint_from_colour_ids() {
        // A pipe tag encodes with the high bit set; the low bits recover the tag.
        for tag in [0u16, 1, 7, 42, PIPE_TAG_MASK] {
            let sd = encode_pipe_side_data(tag);
            assert!(sd & PIPE_TAG_FLAG != 0, "flag set");
            assert_eq!(decode_pipe_tag(sd), Some(tag), "tag round-trips");
        }
        // Plain geometry side_data (0 = none, small colour ids) is NOT a pipe cell,
        // so the colour path is untouched — back-compat.
        for colour_id in [0u16, 1, 2, 5, 0x7FFF] {
            assert_eq!(
                decode_pipe_tag(colour_id),
                None,
                "colour id is not a pipe tag"
            );
        }
    }

    #[test]
    fn inert_object_tag_is_disjoint_from_pipe_node_and_colour_namespaces() {
        // Not a pipe: the pipe decoder must never see an inert bench as a trace.
        assert_eq!(decode_pipe_tag(INERT_OBJECT_TAG), None);
        assert_eq!(INERT_OBJECT_TAG & PIPE_TAG_FLAG, 0);
        // Not a node index / colour id (those count up from 1 and stay below the
        // reserved band): pinned at compile time by the `const _` asserts beside it.
        // No pipe encoding can produce it either.
        for tag in [0u16, 3, 0x7F03, PIPE_TAG_MASK] {
            assert_ne!(encode_pipe_side_data(tag), INERT_OBJECT_TAG);
        }
    }

    #[test]
    fn holo_styles_decode_presets_and_authored_colours() {
        assert_eq!(decode_holo_style(0), None, "0 is the default blue");
        assert_eq!(holo_style_named("Magenta"), Some(5));
        assert_eq!(decode_holo_style(5), Some(0x00FF_48E0));
        assert_eq!(holo_style_named("plaid"), None);
        let s = encode_holo_rgb(0x00FF_40C0);
        assert_eq!(s, 0x1F4B);
        assert_eq!(decode_holo_style(s), Some(0x00FF_44BB));
        // A style past the presets but outside the RGB band is reserved: default.
        assert_eq!(decode_holo_style(HOLO_PRESETS.len() as u16 + 1), None);
        assert_eq!(decode_holo_style(0x2000), None);
    }

    /// Every reserved sentinel (the door marker, the inert bench, ...) is disjoint
    /// from the pipe namespace, from every other sentinel, and from anything a pipe
    /// encoding can produce. This is the test the old `0xD002` door tag would fail.
    #[test]
    fn reserved_sentinels_are_disjoint_from_pipes_and_each_other() {
        for (i, &s) in RESERVED_SENTINELS.iter().enumerate() {
            assert_eq!(decode_pipe_tag(s), None, "{s:#06x} decodes as a pipe");
            assert!(
                (RESERVED_TAG_BASE..=PIPE_TAG_MASK).contains(&s),
                "{s:#06x} outside the band"
            );
            for &o in &RESERVED_SENTINELS[i + 1..] {
                assert_ne!(s, o, "two sentinels share {s:#06x}");
            }
            for tag in [0u16, 1, s & PIPE_TAG_MASK, 0x5002, PIPE_TAG_MASK] {
                assert_ne!(
                    encode_pipe_side_data(tag),
                    s,
                    "pipe {tag:#06x} encodes to sentinel {s:#06x}"
                );
            }
        }
        assert_eq!(DOOR_OBJECT_TAG, 0x7F02);
        assert_eq!(
            decode_pipe_tag(0xD002),
            Some(0x5002),
            "why the old door tag had to move"
        );
    }

    #[test]
    fn flow_state_round_trips_each_direction() {
        for dir in [FlowDir::None, FlowDir::Forward, FlowDir::Backward] {
            assert_eq!(
                decode_flow_state(encode_flow_state(dir)),
                dir,
                "{dir:?} round-trips"
            );
        }
        // The empty-bus default (state 0) is an unpowered, non-flowing pipe.
        assert_eq!(decode_flow_state(0), FlowDir::None, "state 0 = cold/static");
        assert_eq!(
            encode_flow_state(FlowDir::None),
            0,
            "no-flow encodes to the empty default"
        );
        // A powered word is distinguishable from the cold default.
        assert_ne!(encode_flow_state(FlowDir::Forward), 0);
        assert_ne!(
            encode_flow_state(FlowDir::Forward),
            encode_flow_state(FlowDir::Backward)
        );
    }
}
