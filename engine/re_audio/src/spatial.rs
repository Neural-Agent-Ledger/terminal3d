//! `spatial` — positional sound sources heard from the player's position and
//! facing. **Pure logic, no device.**
//!
//! The game describes the world's *sounding things* — lamps, powered pipes, lit
//! nodes, particle emitters — as [`PositionalSource`]s, and the player's ears as
//! a [`Listener`] (camera position + yaw, in world cells). This module decides,
//! deterministically, how each source is heard:
//!
//! * [`spatialize`] — distance attenuation (the same `1/(1+d²/r²)` law the
//!   renderer uses for point lights, so a lamp sounds as far away as it looks),
//!   an equal-power pan from the source's bearing relative to the player's yaw
//!   (turn your head and the hum moves across the stereo field), a mild
//!   "behind you" attenuation, **air absorption** (distant sources lose their
//!   top end before they lose their loudness — [`air_absorption_cutoff_hz`]),
//!   **occlusion** (a wall between the source and the listener both quietens
//!   and muffles it — [`PositionalSource::occlusion`], computed by the game
//!   side via a grid ray test; this module only applies the number), a
//!   **front/back cue** (a further slight lowpass on anything behind the
//!   player, on top of the gain cut — no HRTF, but an honest, cheap cue), and
//!   an **ITD** ([`itd_ms`]) for lateral sources: the far ear hears a source a
//!   fraction of a millisecond later than the near ear, the way real ears do.
//! * [`select_voices`] — the voice budget: only the `max` most audible sources
//!   play, ordered by audibility with a stable tie-break, so a room full of
//!   lamps does not become a wall of hum and the choice never flickers.
//!
//! Coordinate convention (from `re_world::CameraPose::dir`): facing is
//! `(sin yaw, cos yaw)` on the x/y floor plane; the player's right is
//! `(cos yaw, −sin yaw)` (the gamepad strafe vector).
//!
//! What's here is deliberately what fits a "pure function of distance/angle/
//! occlusion" shape: **wet/dry-by-distance** is not modeled as an explicit
//! per-voice send level here — it falls out for free once a voice's *dry*
//! gain already obeys inverse-square attenuation and is routed through the
//! shared local-room reverb (`crate::playback`): the reverberant tail stays
//! roughly constant across the room while the dry signal keeps dropping, so
//! the wet:dry ratio rises with distance exactly as in a real room, with no
//! extra per-voice mix parameter to get wrong. **Doppler** is deliberately not
//! modeled (see `docs/*` follow-up note) — positional sources here are mostly
//! static fixtures (lamps, pipes, nodes); the one moving emitter class did not
//! justify the added per-voice velocity bookkeeping this pass.

/// The player's ears: world position (cells) + yaw (radians).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Listener {
    /// World position `[x, y, z]` in cells.
    pub pos: [f32; 3],
    /// Facing yaw in radians (`re_world::CameraPose::yaw`).
    pub yaw: f32,
}

/// What kind of looping voice a source plays. Each kind has its own seamless
/// loop rendered by [`crate::soundscape::render_loop`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LoopKind {
    /// A light fixture: mains buzz + filament crackle.
    LampHum,
    /// A powered pipe: flowing current. Pitch and level follow the power load.
    PipeHum,
    /// A lit (firing) node: a soft sustained tone at the node's pitch.
    NodeGlow,
    /// A particle emitter (steam, sparks): a breathy hiss.
    EmitterHiss,
}

impl LoopKind {
    /// Every kind, in canonical order.
    pub const ALL: [LoopKind; 4] = [
        LoopKind::LampHum,
        LoopKind::PipeHum,
        LoopKind::NodeGlow,
        LoopKind::EmitterHiss,
    ];

    /// Index into [`LoopKind::ALL`].
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Default audible radius (cells) — the distance at which the source is at
    /// half strength.
    pub fn radius(self) -> f32 {
        match self {
            LoopKind::LampHum => 3.0,
            LoopKind::PipeHum => 2.5,
            LoopKind::NodeGlow => 3.5,
            LoopKind::EmitterHiss => 4.0,
        }
    }
}

/// A stable identity for a source across frames (so a voice keeps playing
/// rather than restarting when the list is rebuilt).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceKey {
    /// The loop the source plays.
    pub kind: LoopKind,
    /// Game-assigned id, unique per kind (light index, pipe index, …).
    pub id: u32,
}

/// One sounding thing in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionalSource {
    /// Stable identity.
    pub key: SourceKey,
    /// World position `[x, y, z]` in cells.
    pub pos: [f32; 3],
    /// Source strength in `[0, 1]` (e.g. a pipe's share of the power load).
    pub level: f32,
    /// Playback rate (pitch) — 1.0 = the loop as rendered.
    pub rate: f64,
    /// How much wall sits between this source and the listener right now, in
    /// `[0, 1]` (0 = clear line of sight, 1 = fully behind solid geometry).
    /// Computed by the host game via a grid ray test; `re_audio` only
    /// applies it (extra attenuation + muffling in [`spatialize`]). Defaults
    /// to `0.0` for callers that don't probe occlusion (offline tests, examples).
    pub occlusion: f32,
}

/// How a source is heard right now.
///
/// Carries two parallel readings on purpose. The device path
/// (`crate::playback`) hands 3D positions to a real **kira spatial track +
/// listener**, so kira itself computes distance attenuation and panning on
/// the audio thread — this module must not duplicate that. What kira cannot
/// know is OUR world (a wall between source and listener, which way the
/// player is facing), so those effects are still computed here and applied
/// as a SEPARATE, near-baseline multiplier the device layers on top of
/// kira's own attenuation via the source's OWN track volume and a second,
/// Rust-driven filter. The OFFLINE mixdown (`crate::offline`, no kira, no
/// device) has no such split to make — it uses the fully combined fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spatial {
    /// Fully combined linear gain in `[0, 1]` (distance × occlusion ×
    /// front/back). **Offline path only** — the device path gets distance
    /// from kira and only needs [`Spatial::near_gain`].
    pub gain: f32,
    /// Stereo position `[-1, 1]` (−1 = hard left). **Offline path only** —
    /// the device path gets panning from kira's own listener/track geometry.
    pub pan: f32,
    /// Fully combined low-pass cutoff in Hz (air absorption × occlusion ×
    /// front/back). **Offline path only**, applied as the single filter in
    /// `crate::offline::OfflineMix::add_positional`.
    pub cutoff_hz: f32,
    /// Occlusion × front/back ONLY, as a multiplier on top of whatever
    /// distance-based gain kira's spatial track already applied (no distance
    /// term — kira owns that). `1.0` = no extra reduction (clear line of
    /// sight, dead ahead).
    pub near_gain: f32,
    /// Occlusion × front/back ONLY, baseline [`AIR_NEAR_HZ`] (no distance
    /// term — kira's own [`crate::playback`] air-absorption filter, bound to
    /// listener distance, is meant to be chained BEFORE this one). Device
    /// path only.
    pub near_cutoff_hz: f32,
}

/// A selected, spatialized voice ready for the device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceMix {
    /// Which source this mix belongs to.
    pub key: SourceKey,
    /// See [`Spatial::gain`] — offline path.
    pub gain: f32,
    /// See [`Spatial::pan`] — offline path.
    pub pan: f32,
    /// Playback rate (pitch): 1.0 = as rendered, 2.0 = an octave up.
    pub rate: f64,
    /// See [`Spatial::cutoff_hz`] — offline path.
    pub cutoff_hz: f32,
    /// Interaural time difference in milliseconds for this voice's `pan`
    /// (see [`itd_ms`]) — positive delays the LEFT channel (source on the
    /// right), negative delays the RIGHT channel (source on the left).
    /// Offline path only — no kira equivalent (see the module doc).
    pub itd_ms: f32,
    /// Raw world position (cells) — device path: handed to kira's spatial
    /// track, which computes its own distance attenuation + panning.
    pub pos: [f32; 3],
    /// The source's own intrinsic level (`PositionalSource::level`, not
    /// distance-attenuated) — device path: the instance's own base volume,
    /// on top of which kira applies distance and we apply [`Self::near_gain`].
    pub source_level: f32,
    /// See [`Spatial::near_gain`] — device path.
    pub near_gain: f32,
    /// See [`Spatial::near_cutoff_hz`] — device path.
    pub near_cutoff_hz: f32,
}

/// Maximum pan magnitude: a positional source is never pinned hard into one
/// ear (that reads as headphones-broken, not as "over there").
pub const MAX_PAN: f32 = 0.9;

/// Beyond `CUTOFF_RADII × radius` a source is silent (the tail of the
/// attenuation law is faded to exactly zero there, so voices actually stop).
pub const CUTOFF_RADII: f32 = 4.0;

// ───────────────────────────── air absorption ────────────────────────────────

/// Distance (cells) at and below which a source is heard full-band — real air
/// absorption is negligible over a few metres.
pub const AIR_NEAR_CELLS: f32 = 1.5;
/// Cutoff (Hz) at [`AIR_NEAR_CELLS`] and closer — effectively "open".
pub const AIR_NEAR_HZ: f32 = 17_000.0;
/// Cutoff (Hz) a source is filtered to by the time it reaches the edge of its
/// audible radius — high frequencies are gone well before the source falls
/// silent, which is what real distant sources sound like (a muffled thump,
/// not a quiet full-band copy).
pub const AIR_FAR_HZ: f32 = 900.0;

/// Air-absorption low-pass cutoff (Hz) for a source `distance_cells` away
/// with audible `radius` (the same radius [`spatialize`] attenuates by).
/// Monotonically falls from [`AIR_NEAR_HZ`] to [`AIR_FAR_HZ`] between
/// [`AIR_NEAR_CELLS`] and the audible cutoff distance.
pub fn air_absorption_cutoff_hz(distance_cells: f32, radius: f32) -> f32 {
    let far_edge = radius.max(0.1) * CUTOFF_RADII;
    let span = (far_edge - AIR_NEAR_CELLS).max(0.1);
    let t = ((distance_cells - AIR_NEAR_CELLS) / span).clamp(0.0, 1.0);
    AIR_NEAR_HZ + (AIR_FAR_HZ - AIR_NEAR_HZ) * t
}

// ───────────────────────────── occlusion ─────────────────────────────────────

/// A fully occluded source is still audible (through-the-wall thump), just at
/// this fraction of its clear-line-of-sight gain.
pub const OCCLUSION_GAIN_FLOOR: f32 = 0.35;
/// A fully occluded source's cutoff never goes above this (Hz) — a heavy,
/// muffled-through-a-wall lowpass, independent of how open its air-absorption
/// cutoff would otherwise be (a loud close source right behind a thin wall
/// should still sound covered, not merely "distant").
pub const OCCLUSION_CUTOFF_CEILING_HZ: f32 = 1400.0;

fn apply_occlusion_gain(gain: f32, occlusion: f32) -> f32 {
    let o = occlusion.clamp(0.0, 1.0);
    gain * (1.0 - (1.0 - OCCLUSION_GAIN_FLOOR) * o)
}

fn apply_occlusion_cutoff(cutoff_hz: f32, occlusion: f32) -> f32 {
    let o = occlusion.clamp(0.0, 1.0);
    cutoff_hz.min(cutoff_hz * (1.0 - o) + OCCLUSION_CUTOFF_CEILING_HZ * o)
}

// ───────────────────────────── front/back + ITD ──────────────────────────────

/// The cutoff multiplier applied when a source is directly behind the player
/// (`ahead == -1`); directly ahead (`ahead == 1`) applies no reduction. A
/// "slight" lowpass, as specified — audible but not a dramatic mute.
const BEHIND_CUTOFF_MIN_FACTOR: f32 = 0.55;

/// Maximum interaural time difference in milliseconds: how much later the far
/// ear hears a fully-lateral (`pan == ±1`) source than the near ear. Real
/// human ITD tops out around 0.6-0.7 ms; used here as a linear function of
/// pan (not physically a head-shadow model, but the right sign and the right
/// order of magnitude — audible as "over there" on headphones without being a
/// gimmick).
pub const MAX_ITD_MS: f32 = 0.65;

/// Interaural time difference (ms) for a voice at `pan` (`[-1, 1]`, see
/// [`Spatial::pan`]). Positive delays the LEFT channel (source on the right,
/// so the right/near ear hears it first); negative delays the RIGHT channel.
#[inline]
pub fn itd_ms(pan: f32) -> f32 {
    pan.clamp(-1.0, 1.0) * MAX_ITD_MS
}

/// Hear a point source at `src` with audible `radius` from `l`, `occlusion`
/// in `[0, 1]` (0 = clear line of sight).
pub fn spatialize(l: &Listener, src: [f32; 3], radius: f32, occlusion: f32) -> Spatial {
    let dx = src[0] - l.pos[0];
    let dy = src[1] - l.pos[1];
    let dz = src[2] - l.pos[2];
    let d2 = dx * dx + dy * dy + dz * dz;
    let d = d2.sqrt();
    let r = radius.max(0.1);
    let cutoff_dist = r * CUTOFF_RADII;
    if d >= cutoff_dist {
        return Spatial {
            gain: 0.0,
            pan: 0.0,
            cutoff_hz: AIR_FAR_HZ,
            near_gain: 0.0,
            near_cutoff_hz: AIR_FAR_HZ,
        };
    }
    // Light-matching attenuation, faded linearly to zero over the last radius.
    let mut gain = 1.0 / (1.0 + d2 / (r * r));
    let fade_start = cutoff_dist - r;
    if d > fade_start {
        gain *= (cutoff_dist - d) / r;
    }

    let (fx, fy) = (l.yaw.sin(), l.yaw.cos());
    let (rx, ry) = (l.yaw.cos(), -l.yaw.sin());
    let h = (dx * dx + dy * dy).sqrt();
    let (pan, ahead) = if h > 1e-4 {
        ((dx * rx + dy * ry) / h, (dx * fx + dy * fy) / h)
    } else {
        (0.0, 1.0)
    };
    // Very close sources collapse toward the centre (inside your head), so
    // walking *through* a source never flips it hard left→right in one frame.
    let near = (h / 0.75).min(1.0);
    // Behind the player: up to 25 % quieter (no HRTF, but the cue is honest).
    let behind_gain_factor = 0.875 + 0.125 * ahead;
    gain *= behind_gain_factor;
    gain = apply_occlusion_gain(gain, occlusion);

    // `near_gain`: the SAME occlusion + front/back reduction, but with NO
    // distance term — kira's own spatial-track attenuation already covers
    // distance on the device path (see the module doc), so this is a
    // multiplier centred on "no reduction" rather than a second attenuation
    // curve layered on top of kira's.
    let near_gain = apply_occlusion_gain(behind_gain_factor, occlusion);

    let mut cutoff = air_absorption_cutoff_hz(d, r);
    // Front/back: a further slight lowpass behind, on top of the air cutoff.
    let behind_cutoff_factor =
        BEHIND_CUTOFF_MIN_FACTOR + (1.0 - BEHIND_CUTOFF_MIN_FACTOR) * (0.5 + 0.5 * ahead);
    cutoff *= behind_cutoff_factor;
    cutoff = apply_occlusion_cutoff(cutoff, occlusion);

    // `near_cutoff_hz`: baseline AIR_NEAR_HZ (no distance term — the device
    // path chains kira's own listener-distance-bound air filter BEFORE this
    // one, see `crate::playback`), just occlusion + front/back.
    let near_cutoff_hz = apply_occlusion_cutoff(AIR_NEAR_HZ * behind_cutoff_factor, occlusion);

    Spatial {
        gain: gain.clamp(0.0, 1.0),
        pan: (pan * near).clamp(-MAX_PAN, MAX_PAN),
        cutoff_hz: cutoff.clamp(200.0, AIR_NEAR_HZ),
        near_gain: near_gain.clamp(0.0, 1.0),
        near_cutoff_hz: near_cutoff_hz.clamp(200.0, AIR_NEAR_HZ),
    }
}

/// Pick the `max` most audible sources and spatialize them. Deterministic:
/// ordered by audibility (gain × level) descending, ties broken by key. Sources
/// quieter than `0.005` are dropped entirely.
pub fn select_voices(l: &Listener, sources: &[PositionalSource], max: usize) -> Vec<VoiceMix> {
    let mut v: Vec<VoiceMix> = sources
        .iter()
        .filter_map(|s| {
            let sp = spatialize(l, s.pos, s.key.kind.radius(), s.occlusion);
            let gain = sp.gain * s.level.clamp(0.0, 1.0);
            (gain >= 0.005).then_some(VoiceMix {
                key: s.key,
                gain,
                pan: sp.pan,
                rate: s.rate,
                cutoff_hz: sp.cutoff_hz,
                itd_ms: itd_ms(sp.pan),
                pos: s.pos,
                source_level: s.level.clamp(0.0, 1.0),
                near_gain: sp.near_gain,
                near_cutoff_hz: sp.near_cutoff_hz,
            })
        })
        .collect();
    v.sort_by(|a, b| b.gain.total_cmp(&a.gain).then(a.key.cmp(&b.key)));
    v.truncate(max);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at_origin_facing_north() -> Listener {
        // yaw 0 → facing +y, right is +x.
        Listener {
            pos: [0.0, 0.0, 1.5],
            yaw: 0.0,
        }
    }

    #[test]
    fn a_source_on_the_right_pans_right_and_left_pans_left() {
        let l = at_origin_facing_north();
        let right = spatialize(&l, [2.0, 0.0, 1.5], 3.0, 0.0);
        let left = spatialize(&l, [-2.0, 0.0, 1.5], 3.0, 0.0);
        let ahead = spatialize(&l, [0.0, 2.0, 1.5], 3.0, 0.0);
        assert!(right.pan > 0.5, "{right:?}");
        assert!(left.pan < -0.5, "{left:?}");
        assert!(ahead.pan.abs() < 1e-4, "{ahead:?}");
        assert!((right.gain - left.gain).abs() < 1e-5);
    }

    #[test]
    fn turning_the_head_moves_the_source_across_the_field() {
        let src = [0.0, 3.0, 1.5]; // due north
        let facing_north = spatialize(
            &Listener {
                pos: [0.0, 0.0, 1.5],
                yaw: 0.0,
            },
            src,
            3.0,
            0.0,
        );
        // Turn 90° right (yaw +π/2 → facing +x): north is now on the LEFT.
        let facing_east = spatialize(
            &Listener {
                pos: [0.0, 0.0, 1.5],
                yaw: std::f32::consts::FRAC_PI_2,
            },
            src,
            3.0,
            0.0,
        );
        assert!(facing_north.pan.abs() < 1e-4);
        assert!(facing_east.pan < -0.5, "{facing_east:?}");
    }

    #[test]
    fn distance_attenuates_monotonically_and_cuts_off() {
        let l = at_origin_facing_north();
        let mut prev = f32::INFINITY;
        for d in [0.5f32, 1.0, 2.0, 4.0, 8.0, 11.0] {
            let g = spatialize(&l, [0.0, d, 1.5], 3.0, 0.0).gain;
            assert!(g < prev, "gain did not fall at {d}: {g} >= {prev}");
            prev = g;
        }
        assert_eq!(spatialize(&l, [0.0, 12.0, 1.5], 3.0, 0.0).gain, 0.0);
        // At d == radius the attenuation law gives half strength (ahead).
        let g = spatialize(&l, [0.0, 3.0, 1.5], 3.0, 0.0).gain;
        assert!((g - 0.5).abs() < 1e-4, "{g}");
    }

    #[test]
    fn behind_is_quieter_and_more_muffled_than_ahead() {
        let l = at_origin_facing_north();
        let ahead = spatialize(&l, [0.0, 2.0, 1.5], 3.0, 0.0);
        let behind = spatialize(&l, [0.0, -2.0, 1.5], 3.0, 0.0);
        assert!(behind.gain < ahead.gain);
        assert!(behind.gain > 0.7 * ahead.gain);
        assert!(
            behind.cutoff_hz < ahead.cutoff_hz,
            "behind should be darker: {} vs {}",
            behind.cutoff_hz,
            ahead.cutoff_hz
        );
    }

    #[test]
    fn distant_sources_lose_their_top_end_before_they_lose_their_loudness() {
        let l = at_origin_facing_north();
        let near = spatialize(&l, [0.0, 0.5, 1.5], 3.0, 0.0);
        let mid = spatialize(&l, [0.0, 4.0, 1.5], 3.0, 0.0);
        let far = spatialize(&l, [0.0, 9.0, 1.5], 3.0, 0.0);
        assert!(
            near.cutoff_hz > mid.cutoff_hz && mid.cutoff_hz > far.cutoff_hz,
            "{near:?} {mid:?} {far:?}"
        );
        // Air absorption is audible well before the source has faded to
        // silence: at `mid` there is still real gain left, but the cutoff has
        // already moved well down from the near ceiling.
        assert!(mid.gain > 0.05);
        assert!(mid.cutoff_hz < AIR_NEAR_HZ * 0.85);
    }

    #[test]
    fn occlusion_quietens_and_muffles_without_ever_fully_silencing() {
        let l = at_origin_facing_north();
        let clear = spatialize(&l, [0.0, 2.0, 1.5], 3.0, 0.0);
        let occluded = spatialize(&l, [0.0, 2.0, 1.5], 3.0, 1.0);
        assert!(occluded.gain < clear.gain);
        assert!(
            occluded.gain > 0.0,
            "a fully occluded source must still be audible (through-the-wall)"
        );
        assert!(occluded.cutoff_hz < clear.cutoff_hz);
        assert!(occluded.cutoff_hz <= OCCLUSION_CUTOFF_CEILING_HZ + 1e-3);
        // Monotone in between.
        let half = spatialize(&l, [0.0, 2.0, 1.5], 3.0, 0.5);
        assert!(half.gain < clear.gain && half.gain > occluded.gain);
        assert!(half.cutoff_hz < clear.cutoff_hz && half.cutoff_hz > occluded.cutoff_hz);
    }

    #[test]
    fn itd_matches_the_pan_sign_and_is_bounded() {
        assert_eq!(itd_ms(0.0), 0.0);
        assert!(itd_ms(1.0) > 0.0 && (itd_ms(1.0) - MAX_ITD_MS).abs() < 1e-5);
        assert!(itd_ms(-1.0) < 0.0 && (itd_ms(-1.0) + MAX_ITD_MS).abs() < 1e-5);
        assert!(itd_ms(2.0) <= MAX_ITD_MS + 1e-5, "must clamp beyond ±1");
        // Monotonic in pan.
        assert!(itd_ms(0.3) < itd_ms(0.9));
    }

    #[test]
    fn near_gain_and_near_cutoff_have_no_distance_term() {
        // Same angle (dead ahead) and occlusion, wildly different distance:
        // near_gain / near_cutoff_hz must match, because kira's own
        // spatial-track distance attenuation owns the distance axis on the
        // device path — this module must not double it.
        let l = at_origin_facing_north();
        let near = spatialize(&l, [0.0, 0.5, 1.5], 3.0, 0.3);
        let far = spatialize(&l, [0.0, 8.0, 1.5], 3.0, 0.3);
        assert!(
            (near.near_gain - far.near_gain).abs() < 1e-5,
            "{near:?} {far:?}"
        );
        assert!(
            (near.near_cutoff_hz - far.near_cutoff_hz).abs() < 1e-3,
            "{near:?} {far:?}"
        );
        // But the fully combined (offline-path) fields DO still fall off.
        assert!(far.gain < near.gain);
        assert!(far.cutoff_hz < near.cutoff_hz);
    }

    #[test]
    fn near_gain_and_near_cutoff_still_react_to_occlusion_and_facing() {
        let l = at_origin_facing_north();
        let ahead_clear = spatialize(&l, [0.0, 2.0, 1.5], 3.0, 0.0);
        let behind_clear = spatialize(&l, [0.0, -2.0, 1.5], 3.0, 0.0);
        let ahead_occluded = spatialize(&l, [0.0, 2.0, 1.5], 3.0, 1.0);
        assert!(behind_clear.near_gain < ahead_clear.near_gain);
        assert!(ahead_occluded.near_gain < ahead_clear.near_gain);
        assert!(ahead_occluded.near_gain > 0.0, "must never fully silence");
        assert!(behind_clear.near_cutoff_hz < ahead_clear.near_cutoff_hz);
        assert!(ahead_occluded.near_cutoff_hz < ahead_clear.near_cutoff_hz);
    }

    #[test]
    fn select_voices_keeps_the_loudest_deterministically() {
        let l = at_origin_facing_north();
        let mk = |id: u32, y: f32| PositionalSource {
            key: SourceKey {
                kind: LoopKind::LampHum,
                id,
            },
            pos: [0.0, y, 1.5],
            level: 1.0,
            rate: 1.0,
            occlusion: 0.0,
        };
        let sources = [mk(0, 9.0), mk(1, 1.0), mk(2, 3.0), mk(3, 2.0), mk(4, 50.0)];
        let v = select_voices(&l, &sources, 3);
        let ids: Vec<u32> = v.iter().map(|m| m.key.id).collect();
        assert_eq!(ids, vec![1, 3, 2]);
        // Silent / out-of-range sources are dropped even when budget remains.
        let v = select_voices(&l, &sources, 10);
        assert!(v.iter().all(|m| m.key.id != 4));
        // A zero-level source (an unpowered pipe) is not a voice.
        let dead = PositionalSource {
            level: 0.0,
            ..mk(9, 1.0)
        };
        assert!(select_voices(&l, &[dead], 4).is_empty());
    }

    #[test]
    fn select_voices_carries_cutoff_and_itd_through() {
        let l = at_origin_facing_north();
        let far_occluded = PositionalSource {
            key: SourceKey {
                kind: LoopKind::PipeHum,
                id: 0,
            },
            pos: [2.5, 5.0, 1.5],
            level: 1.0,
            rate: 1.0,
            occlusion: 0.8,
        };
        let v = select_voices(&l, &[far_occluded], 4);
        assert_eq!(v.len(), 1);
        assert!(v[0].cutoff_hz < AIR_NEAR_HZ);
        assert!(
            v[0].itd_ms > 0.0,
            "source is to the right, left ear should lag"
        );
        assert!(
            v[0].near_gain < 1.0,
            "occluded source must carry a sub-1 near_gain"
        );
        assert!(v[0].near_cutoff_hz < AIR_NEAR_HZ);
        assert_eq!(v[0].pos, far_occluded.pos);
        assert_eq!(v[0].source_level, far_occluded.level);
    }
}
