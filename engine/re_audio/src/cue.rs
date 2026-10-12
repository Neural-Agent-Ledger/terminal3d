//! `cue` — the game-event → sound mapping. **Pure logic, no device.**
//!
//! The game no longer asks for *a sound*; it reports *what happened* as a
//! [`Cue`] and this table decides how that event is voiced. Keeping the mapping
//! here (not scattered across the host game's call sites) is what makes the soundscape
//! a consistent language instead of a pile of blips — and makes it testable.
//!
//! ## The grammar
//!
//! | word | meaning | how |
//! |---|---|---|
//! | **rising** | progress, opened, arrived | glide up / higher playback rate |
//! | **falling** | regress, closed, refused | glide down / lower rate |
//! | **consonant** | correct | partials at 1, 3/2, 2, 5/4 |
//! | **dissonant** | blocked, over budget | minor-second pairs that beat |
//!
//! ## Event table
//!
//! | event | sound | modulation |
//! |---|---|---|
//! | crosshair lands on a switch | `aim_tick` (quiet rising blip) | — |
//! | switch ON | `switch_on` (rising fifth) | pitch sinks as the power budget fills |
//! | switch OFF | `switch_off` (falling fifth, quieter) | — |
//! | press on nothing | `interact_empty` (neutral thud) | — |
//! | refused: no power | `refuse_power` (dissonant brownout) | + harmonic tension spike |
//! | refused: loop | `refuse_cycle` (stuttering clash, accent track) | + tension spike |
//! | node lit | `node_lit` (chime scooping up) | climbs a major-pentatonic ladder with lit/total, panned from the node |
//! | node went dark | `node_unlit` (falling, m2 shadow) | ladder position of what is left lit, + small tension |
//! | network satisfied | `circuit_resolve` (A-major bloom) | — |
//! | network broken again | `node_unlit` low + `refuse_power` whisper | + tension |
//! | footstep | per surface: `footstep` / `_metal` / `_wet` / `_soft` | alternating feet ±3 % pitch, ±6 % pan |
//! | note in reading range | `note_read` | rate per note kind (log = brighter, model thought = lower) |
//! | portal crossed | `portal_warp` | — |
//! | storey change | `storey_up` / `storey_down` | — |
//! | colour shard (First Light) | `first_light` bloom | each later shard a whole tone higher |
//! | menu move / confirm / back | `ui_move` / `ui_select` / `ui_move` low | — |
//! | level won | `win_sting` + `door` | — |
//!
//! The harmonic side of the grammar — the ambience chord that fills in as the
//! network lights — lives in [`crate::soundscape::harmony_gains`]; the cues here
//! only *spike* its tension through [`Cue::tension`].

use crate::synth::SoundId;

/// What the player is standing on (chooses the footstep recipe).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Surface {
    /// Concrete / stone / tile (the default).
    #[default]
    Hard,
    /// Grating, plate, circuitry.
    Metal,
    /// Puddled / rotting / molten.
    Wet,
    /// Snow, frost, moss, organic.
    Soft,
}

impl Surface {
    /// Classify a material name or catalogue preset (`metal`, `wet_stone`,
    /// `frost`, …). Unknown names are [`Surface::Hard`].
    pub fn from_material(name: &str) -> Surface {
        let n = name.to_ascii_lowercase();
        let has = |keys: &[&str]| keys.iter().any(|k| n.contains(k));
        if has(&["wet", "decay", "lava", "water", "sludge", "rot", "puddle"]) {
            Surface::Wet
        } else if has(&[
            "snow", "frost", "ice", "moss", "grass", "organic", "flesh", "carpet", "soil",
        ]) {
            Surface::Soft
        } else if has(&[
            "metal", "grate", "gold", "neon", "monitor", "steel", "iron", "circuit", "trace",
            "chrome",
        ]) {
            Surface::Metal
        } else {
            Surface::Hard
        }
    }
}

/// The kind of placard read (mirrors `re_level::NoteKind` without the dep).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoteVoice {
    /// A human-authored research page — reads at the neutral rate.
    Researcher,
    /// A machine/system log entry — brighter, more digital.
    SystemLog,
    /// An interior model "thought" — lower, more inward.
    ModelThought,
    /// Plain environmental signage.
    Stencil,
}

/// A game event, as the audio layer needs to know it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cue {
    /// The crosshair just landed on an interactable switch.
    AimSwitch,
    /// A pipe switched ON. `used`/`budget` is the power readout AFTER the
    /// toggle; `budget == 0` means unlimited.
    SwitchOn {
        /// Power units in use after this toggle.
        used: u32,
        /// Power budget; `0` means unlimited.
        budget: u32,
    },
    /// A pipe switched OFF.
    SwitchOff,
    /// Interact pressed with nothing in front.
    InteractEmpty,
    /// Refused: the power budget is spent.
    RefusedNoPower,
    /// Refused: the toggle would close a forbidden loop.
    RefusedCycle,
    /// A node started firing; `lit` of `total` nodes now fire.
    NodeLit {
        /// Nodes currently firing.
        lit: u32,
        /// Total nodes in the network.
        total: u32,
    },
    /// A node went dark; `lit` of `total` still fire.
    NodeUnlit {
        /// Nodes still firing after this one went dark.
        lit: u32,
        /// Total nodes in the network.
        total: u32,
    },
    /// The network's win condition just became satisfied.
    CircuitComplete,
    /// The network was satisfied and no longer is.
    CircuitBroken,
    /// One footstep; `stride` counts steps (parity = which foot).
    Footstep {
        /// Ground material under the foot.
        surface: Surface,
        /// Step counter; parity picks left/right foot voicing.
        stride: u32,
    },
    /// A placard came into reading range.
    NoteRead(NoteVoice),
    /// Crossed a non-euclidean portal.
    Portal,
    /// Moved to another storey.
    Storey {
        /// `true` if the move was upward.
        up: bool,
    },
    /// A colour shard was recovered; `shards` is the new total (1 = First Light).
    FirstLight {
        /// Total shards recovered so far, including this one.
        shards: u32,
    },
    /// Menu highlight moved.
    MenuMove,
    /// Menu item confirmed.
    MenuConfirm,
    /// Menu back / cancel.
    MenuBack,
    /// Level won (network satisfied and the player at the door).
    Win,
}

/// One voiced layer of a cue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voice {
    /// Which pre-rendered sound to play.
    pub sound: SoundId,
    /// Playback rate (pitch): 1.0 = as rendered, 2.0 = an octave up.
    pub rate: f64,
    /// Linear gain in `[0, 1]`.
    pub gain: f32,
    /// Stereo position offset `[-1, 1]` (added to any positional pan).
    pub pan: f32,
}

const fn v(sound: SoundId, rate: f64, gain: f32, pan: f32) -> Voice {
    Voice {
        sound,
        rate,
        gain,
        pan,
    }
}

/// Audible radius (cells) used to place a positioned cue.
pub const CUE_RADIUS: f32 = 6.0;

/// Semitones of the major-pentatonic ladder the node chime climbs.
const LADDER: [f64; 8] = [0.0, 2.0, 4.0, 7.0, 9.0, 12.0, 14.0, 16.0];

/// Rate for `lit` of `total` on the ladder: the more of the network fires, the
/// higher the chime. `lit == 0` sits at the bottom; `lit == total` at the top.
pub fn ladder_rate(lit: u32, total: u32) -> f64 {
    let total = total.max(1);
    let frac = f64::from(lit.min(total)) / f64::from(total);
    let idx = (frac * (LADDER.len() - 1) as f64).round() as usize;
    2f64.powf(LADDER[idx.min(LADDER.len() - 1)] / 12.0)
}

impl Cue {
    /// The layers that voice this cue (1–2), in play order.
    pub fn voices(self) -> Vec<Voice> {
        use SoundId as S;
        match self {
            Cue::AimSwitch => vec![v(S::AimTick, 1.0, 0.8, 0.0)],
            Cue::SwitchOn { used, budget } => {
                // Still a rising fifth — it IS progress — but it sinks as the
                // budget fills, up to two semitones at the last free slot, so the
                // player hears the grid straining before it refuses.
                let load = if budget == 0 {
                    0.0
                } else {
                    (f64::from(used) / f64::from(budget)).clamp(0.0, 1.0)
                };
                vec![v(S::SwitchOn, 2f64.powf(-2.0 * load / 12.0), 0.9, 0.0)]
            }
            Cue::SwitchOff => vec![v(S::SwitchOff, 1.0, 0.8, 0.0)],
            Cue::InteractEmpty => vec![v(S::InteractEmpty, 1.0, 0.7, 0.0)],
            Cue::RefusedNoPower => vec![v(S::RefusePower, 1.0, 0.9, 0.0)],
            Cue::RefusedCycle => vec![v(S::RefuseCycle, 1.0, 0.9, 0.0)],
            Cue::NodeLit { lit, total } => vec![v(S::NodeLit, ladder_rate(lit, total), 0.85, 0.0)],
            Cue::NodeUnlit { lit, total } => {
                vec![v(S::NodeUnlit, ladder_rate(lit, total), 0.75, 0.0)]
            }
            Cue::CircuitComplete => vec![v(S::CircuitResolve, 1.0, 0.9, 0.0)],
            Cue::CircuitBroken => vec![
                v(S::NodeUnlit, 0.75, 0.8, 0.0),
                v(S::RefusePower, 0.8, 0.35, 0.0),
            ],
            Cue::Footstep { surface, stride } => {
                let sound = match surface {
                    Surface::Hard => S::Footstep,
                    Surface::Metal => S::FootstepMetal,
                    Surface::Wet => S::FootstepWet,
                    Surface::Soft => S::FootstepSoft,
                };
                let left = stride % 2 == 0;
                let (rate, pan) = if left { (0.97, -0.06) } else { (1.03, 0.06) };
                vec![v(sound, rate, 0.55, pan)]
            }
            Cue::NoteRead(kind) => {
                let rate = match kind {
                    NoteVoice::SystemLog => 1.26,    // brighter, digital
                    NoteVoice::Researcher => 1.0,    // a human page
                    NoteVoice::ModelThought => 0.79, // lower, inward
                    NoteVoice::Stencil => 0.89,      // signage: plain
                };
                vec![v(S::NoteRead, rate, 0.8, 0.0)]
            }
            Cue::Portal => vec![v(S::PortalWarp, 1.0, 0.85, 0.0)],
            Cue::Storey { up } => {
                vec![v(
                    if up { S::StoreyUp } else { S::StoreyDown },
                    1.0,
                    0.8,
                    0.0,
                )]
            }
            Cue::FirstLight { shards } => {
                // The first shard is the root; each later one a whole tone up.
                let steps = f64::from(shards.saturating_sub(1).min(6)) * 2.0;
                vec![v(S::FirstLight, 2f64.powf(steps / 12.0), 0.9, 0.0)]
            }
            Cue::MenuMove => vec![v(S::UiMove, 1.0, 0.55, 0.0)],
            Cue::MenuConfirm => vec![v(S::UiSelect, 1.0, 0.7, 0.0)],
            Cue::MenuBack => vec![v(S::UiMove, 0.707, 0.5, 0.0)],
            Cue::Win => vec![v(S::WinSting, 1.0, 1.0, 0.0), v(S::Door, 1.0, 0.8, 0.0)],
        }
    }

    /// The voices placed in the world: when the event has a position (a node
    /// that lit, a switch that refused), each layer is panned toward it from the
    /// listener's point of view and made a little quieter with distance — never
    /// below ~half, because a cue the player caused must stay audible.
    pub fn placed_voices(self, from: Option<([f32; 3], &crate::spatial::Listener)>) -> Vec<Voice> {
        let (g, p) = match from {
            Some((pos, l)) => {
                // Cues are player-caused one-shots (a switch you pressed, a
                // node that just lit) with no wall query of their own — treat
                // them as clear line of sight.
                let s = crate::spatial::spatialize(l, pos, CUE_RADIUS, 0.0);
                (0.45 + 0.55 * s.gain, s.pan)
            }
            None => (1.0, 0.0),
        };
        self.voices()
            .into_iter()
            .map(|v| Voice {
                gain: (v.gain * g).clamp(0.0, 1.0),
                pan: (v.pan + p).clamp(-1.0, 1.0),
                ..v
            })
            .collect()
    }

    /// How much this event spikes the ambience's harmonic TENSION voice
    /// (`[0, 1]`): refusals clash with the room, regress darkens it a little,
    /// everything else leaves it alone.
    pub fn tension(self) -> f32 {
        match self {
            Cue::RefusedNoPower => 1.0,
            Cue::RefusedCycle => 0.85,
            Cue::CircuitBroken => 0.6,
            Cue::NodeUnlit { .. } => 0.25,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(c: Cue) -> Voice {
        c.voices()[0]
    }

    #[test]
    fn on_and_off_are_different_sounds() {
        assert_eq!(
            first(Cue::SwitchOn { used: 1, budget: 0 }).sound,
            SoundId::SwitchOn
        );
        assert_eq!(first(Cue::SwitchOff).sound, SoundId::SwitchOff);
    }

    #[test]
    fn switch_on_sinks_as_the_budget_fills() {
        let free = first(Cue::SwitchOn { used: 1, budget: 4 }).rate;
        let full = first(Cue::SwitchOn { used: 4, budget: 4 }).rate;
        let unlimited = first(Cue::SwitchOn { used: 9, budget: 0 }).rate;
        assert!(full < free && free < 1.0);
        assert_eq!(unlimited, 1.0);
        // Never more than two semitones: it still reads as the same "on".
        assert!(full > 0.88);
    }

    #[test]
    fn node_chime_climbs_with_progress() {
        let mut prev = 0.0;
        for lit in 0..=6 {
            let r = first(Cue::NodeLit { lit, total: 6 }).rate;
            assert!(r >= prev, "ladder fell at {lit}/6");
            prev = r;
        }
        assert!(ladder_rate(6, 6) > ladder_rate(1, 6) * 1.5);
        assert_eq!(ladder_rate(0, 0), 1.0); // degenerate network: no panic
                                            // Losing a node lands LOWER than the chime that lit it.
        assert!(
            first(Cue::NodeUnlit { lit: 2, total: 6 }).rate
                < first(Cue::NodeLit { lit: 3, total: 6 }).rate
        );
    }

    #[test]
    fn refusals_are_distinct_from_each_other_and_from_acceptance() {
        let p = first(Cue::RefusedNoPower).sound;
        let c = first(Cue::RefusedCycle).sound;
        let ok = first(Cue::SwitchOn { used: 0, budget: 0 }).sound;
        let empty = first(Cue::InteractEmpty).sound;
        let all = [p, c, ok, empty];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j]);
            }
        }
        // Only refusals / regress raise tension.
        assert!(Cue::RefusedNoPower.tension() > 0.5);
        assert!(Cue::RefusedCycle.tension() > 0.5);
        assert_eq!(Cue::SwitchOn { used: 1, budget: 2 }.tension(), 0.0);
        assert_eq!(Cue::CircuitComplete.tension(), 0.0);
    }

    #[test]
    fn footsteps_follow_the_surface_and_alternate_feet() {
        let s = |surface| first(Cue::Footstep { surface, stride: 0 }).sound;
        assert_eq!(s(Surface::Hard), SoundId::Footstep);
        assert_eq!(s(Surface::Metal), SoundId::FootstepMetal);
        assert_eq!(s(Surface::Wet), SoundId::FootstepWet);
        assert_eq!(s(Surface::Soft), SoundId::FootstepSoft);
        let l = first(Cue::Footstep {
            surface: Surface::Hard,
            stride: 0,
        });
        let r = first(Cue::Footstep {
            surface: Surface::Hard,
            stride: 1,
        });
        assert!(l.pan < 0.0 && r.pan > 0.0 && l.rate != r.rate);
    }

    #[test]
    fn a_positioned_cue_comes_from_its_side_and_stays_audible() {
        let l = crate::spatial::Listener {
            pos: [0.0, 0.0, 1.5],
            yaw: 0.0,
        };
        let c = Cue::NodeLit { lit: 1, total: 2 };
        let right = c.placed_voices(Some(([3.0, 0.0, 1.5], &l)))[0];
        let left = c.placed_voices(Some(([-3.0, 0.0, 1.5], &l)))[0];
        let far = c.placed_voices(Some(([0.0, 40.0, 1.5], &l)))[0];
        let unplaced = c.placed_voices(None)[0];
        assert!(right.pan > 0.5 && left.pan < -0.5);
        assert!(far.gain < unplaced.gain && far.gain >= 0.45 * unplaced.gain - 1e-6);
        assert_eq!(unplaced, c.voices()[0]);
    }

    #[test]
    fn material_names_classify() {
        assert_eq!(Surface::from_material("metal"), Surface::Metal);
        assert_eq!(Surface::from_material("wet_stone"), Surface::Wet);
        assert_eq!(Surface::from_material("frost"), Surface::Soft);
        assert_eq!(Surface::from_material("worn_floor"), Surface::Hard);
        assert_eq!(Surface::from_material(""), Surface::Hard);
    }

    #[test]
    fn storeys_and_notes_and_first_light_are_voiced() {
        assert_eq!(first(Cue::Storey { up: true }).sound, SoundId::StoreyUp);
        assert_eq!(first(Cue::Storey { up: false }).sound, SoundId::StoreyDown);
        let log = first(Cue::NoteRead(NoteVoice::SystemLog)).rate;
        let thought = first(Cue::NoteRead(NoteVoice::ModelThought)).rate;
        assert!(
            log > thought,
            "a system log reads brighter than a model thought"
        );
        assert!(
            first(Cue::FirstLight { shards: 3 }).rate > first(Cue::FirstLight { shards: 1 }).rate
        );
        assert_eq!(first(Cue::FirstLight { shards: 1 }).rate, 1.0);
        assert_eq!(Cue::Win.voices().len(), 2);
    }

    #[test]
    fn every_cue_voices_something_audible() {
        let cues = [
            Cue::AimSwitch,
            Cue::SwitchOn { used: 0, budget: 0 },
            Cue::SwitchOff,
            Cue::InteractEmpty,
            Cue::RefusedNoPower,
            Cue::RefusedCycle,
            Cue::NodeLit { lit: 1, total: 3 },
            Cue::NodeUnlit { lit: 0, total: 3 },
            Cue::CircuitComplete,
            Cue::CircuitBroken,
            Cue::Footstep {
                surface: Surface::Soft,
                stride: 3,
            },
            Cue::NoteRead(NoteVoice::Stencil),
            Cue::Portal,
            Cue::Storey { up: true },
            Cue::FirstLight { shards: 7 },
            Cue::MenuMove,
            Cue::MenuConfirm,
            Cue::MenuBack,
            Cue::Win,
        ];
        for c in cues {
            let vs = c.voices();
            assert!(!vs.is_empty(), "{c:?} is silent");
            for voice in vs {
                assert!(
                    voice.gain > 0.0 && voice.gain <= 1.0 && voice.rate > 0.25 && voice.rate < 4.0,
                    "{c:?}"
                );
            }
        }
    }
}
