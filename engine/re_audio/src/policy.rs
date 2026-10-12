//! `policy` — the per-sound quantization table. **Pure logic, no device.**
//!
//! Maps every [`SoundId`] to *how* it should be scheduled against the beat
//! grid: played the instant it is requested ([`Quantize::Immediate`]) or snapped
//! to the next strong/weak tick ([`Quantize::Beat`]). This is the contract the
//! [`crate::plugin`] systems consult before handing a one-shot to
//! [`crate::transport::next_tick`].
//!
//! The table is exhaustive over `SoundId` — the `match` has no wildcard arm, so
//! adding a new sound forces a compile error here until its policy is declared.

use crate::synth::SoundId;
use crate::transport::BeatClass;

/// How a one-shot is scheduled relative to the beat grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quantize {
    /// Play immediately, with zero added latency. Used for player feedback and
    /// motion-tied sounds where any delay would feel wrong. Never humanized.
    Immediate,
    /// Snap to the next tick of the given [`BeatClass`] (strong = bar downbeat,
    /// weak = other beats). Humanize jitter may be applied in `playback`.
    Beat(BeatClass),
}

/// The quantization policy for a sound. Exhaustive over [`SoundId`]: there is
/// deliberately **no** wildcard arm, so a newly-added sound will not compile
/// until it is classified here.
///
/// Per the spec's policy table:
/// * UI / feedback (`Interact`, `UiMove`, `UiSelect`, `WinSting`, `SwitchClick`)
///   and `Footstep` → [`Quantize::Immediate`].
/// * `Door`, `Pipe` and the whole glitch family → `Beat(Weak)`.
/// * `Boot`, drone accents and the tonal palette → `Beat(Strong)`.
/// * Ambient beds are looped, not one-shots; they report `Immediate` here
///   because they are not scheduled through the tick grid at all (the plugin
///   routes them via `SetAmbient`, bypassing quantization).
pub fn quantize_of(id: SoundId) -> Quantize {
    use SoundId::*;
    match id {
        // Player feedback / motion — zero latency.
        Interact | UiMove | UiSelect | WinSting | SwitchClick => Quantize::Immediate,
        Footstep => Quantize::Immediate,

        // Mechanical + glitch family land on weak beats.
        Door | Pipe => Quantize::Beat(BeatClass::Weak),
        Glitch | GlitchTick | DataBurst | Scanline | Stutter => Quantize::Beat(BeatClass::Weak),

        // Boot, tonal palette and drone accents land on strong (down)beats.
        Boot => Quantize::Beat(BeatClass::Strong),
        ToneA3 | ToneC4 | ToneD4 | ToneE4 | ToneG4 | ToneA4 | ToneC5 | ToneD5 => {
            Quantize::Beat(BeatClass::Strong)
        }
        DroneLayerA | DroneLayerB | Voice => Quantize::Beat(BeatClass::Strong),

        // Narrative trigger stingers. The notice ping is a *reaction* to the
        // player crossing a trigger, so it must not lag behind the moment →
        // Immediate. The memory swell is slow and atmospheric: it lands on a bar
        // downbeat. The corruption tick behaves like the rest of the glitch
        // family → weak beat (and routes to the accent track, see below).
        StingerNotice => Quantize::Immediate,
        StingerMemory => Quantize::Beat(BeatClass::Strong),
        StingerCorrupt => Quantize::Beat(BeatClass::Weak),

        // Campaign-movement motifs. All are *musical* material fired by the
        // director on a beat it already classified, so their policy must agree
        // with the beat they are proposed on or they would be re-snapped away
        // from it: the gaze pulse, the river wash and the heartbeat's "lub" land
        // on the bar downbeat; the answering "dub" is the weak beat after it.
        GazePulse | RiverWash | HeartLub => Quantize::Beat(BeatClass::Strong),
        HeartDub => Quantize::Beat(BeatClass::Weak),

        // Ambient beds + atmospheric drones: looped, not tick-scheduled. Reported
        // Immediate but the plugin never routes these through `quantize_of` —
        // they go via the ambience/`SetAmbient` path, crossfaded as a bed.
        Ambient | AmbientCold | AmbientGarden => Quantize::Immediate,
        DroneColdVoid | DroneServerHum | DroneOrganicDecay | DroneAlarmTension => {
            Quantize::Immediate
        }

        // The puzzle sound grammar (see `crate::cue`). Every one of these is the
        // world ANSWERING a player action or a state change the player caused,
        // so the spec's rule for player feedback applies: zero latency. A switch
        // that clicks a quarter-beat after the key press reads as lag, not
        // groove. The single exception is the First Light bloom — a slow,
        // musical swell with no transient, fired by a progress milestone rather
        // than a key press — which lands on the bar downbeat like the other
        // tonal material.
        SwitchOn | SwitchOff | InteractEmpty | RefusePower | RefuseCycle => Quantize::Immediate,
        NodeLit | NodeUnlit | CircuitResolve | AimTick | NoteRead => Quantize::Immediate,
        PortalWarp | StoreyUp | StoreyDown => Quantize::Immediate,
        FootstepMetal | FootstepWet | FootstepSoft => Quantize::Immediate,
        FirstLight => Quantize::Beat(BeatClass::Strong),
    }
}

/// Whether a sound routes through the harsher **accent** track (distortion +
/// short delay) rather than the master reverb/lowpass "fog" path.
///
/// That is the terminal-glitch family ([`SoundId::GLITCH_FAMILY`] + the original
/// `Glitch`) plus [`SoundId::StingerCorrupt`], which is a glitch by texture even
/// though it is a narrative cue and therefore deliberately outside the
/// director's `GLITCH_FAMILY` palette. The other two stingers
/// ([`SoundId::StingerNotice`], [`SoundId::StingerMemory`]) route to the master
/// fog track, so they inherit the theme's reverb/lowpass like every other
/// non-glitch sound.
///
/// [`SoundId::RefuseCycle`] joins them: a forbidden loop is corruption the
/// player almost caused, and the distortion + delay smear is what separates it
/// from the clean (master-fog) power refusal.
pub fn is_glitch_family(id: SoundId) -> bool {
    SoundId::GLITCH_FAMILY.contains(&id)
        || matches!(
            id,
            SoundId::Glitch | SoundId::StingerCorrupt | SoundId::RefuseCycle
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_total_over_every_sound_id() {
        // Exhaustiveness in two senses: the match below compiles only if every
        // variant is handled (no wildcard), and this loop proves `quantize_of`
        // returns for every id in the canonical list without panicking.
        for id in SoundId::ALL {
            let _ = quantize_of(id);
        }
        assert_eq!(SoundId::ALL.len(), crate::synth::SYNTH_SOUND_COUNT);
    }

    #[test]
    fn immediate_sounds_match_spec() {
        for id in [
            SoundId::Interact,
            SoundId::UiMove,
            SoundId::UiSelect,
            SoundId::WinSting,
            SoundId::SwitchClick,
            SoundId::Footstep,
        ] {
            assert_eq!(quantize_of(id), Quantize::Immediate, "{}", id.name());
        }
    }

    #[test]
    fn weak_beat_sounds_match_spec() {
        for id in [
            SoundId::Door,
            SoundId::Pipe,
            SoundId::Glitch,
            SoundId::GlitchTick,
            SoundId::DataBurst,
            SoundId::Scanline,
            SoundId::Stutter,
        ] {
            assert_eq!(
                quantize_of(id),
                Quantize::Beat(BeatClass::Weak),
                "{}",
                id.name()
            );
        }
    }

    #[test]
    fn strong_beat_sounds_match_spec() {
        assert_eq!(
            quantize_of(SoundId::Boot),
            Quantize::Beat(BeatClass::Strong)
        );
        for id in SoundId::TONAL_PALETTE {
            assert_eq!(
                quantize_of(id),
                Quantize::Beat(BeatClass::Strong),
                "{}",
                id.name()
            );
        }
    }

    #[test]
    fn glitch_family_routing() {
        for id in SoundId::GLITCH_FAMILY {
            assert!(is_glitch_family(id), "{}", id.name());
        }
        assert!(is_glitch_family(SoundId::Glitch));
        assert!(!is_glitch_family(SoundId::Boot));
        assert!(!is_glitch_family(SoundId::ToneA3));
    }

    #[test]
    fn stinger_policies_match_the_design() {
        // Notice = player-facing reaction → no latency.
        assert_eq!(quantize_of(SoundId::StingerNotice), Quantize::Immediate);
        // Memory = slow swell → bar downbeat.
        assert_eq!(
            quantize_of(SoundId::StingerMemory),
            Quantize::Beat(BeatClass::Strong)
        );
        // Corruption = glitch texture → weak beat, like the rest of the family.
        assert_eq!(
            quantize_of(SoundId::StingerCorrupt),
            Quantize::Beat(BeatClass::Weak)
        );
    }

    #[test]
    fn movement_motif_policies_match_the_beat_they_are_proposed_on() {
        // The director proposes a motif on a beat it has already classified; the
        // policy must send it to that same class, or `next_tick` would drag it
        // off the beat it was written for.
        for id in [SoundId::GazePulse, SoundId::RiverWash, SoundId::HeartLub] {
            assert_eq!(
                quantize_of(id),
                Quantize::Beat(BeatClass::Strong),
                "{}",
                id.name()
            );
        }
        assert_eq!(
            quantize_of(SoundId::HeartDub),
            Quantize::Beat(BeatClass::Weak)
        );
        // Motifs are musical texture, not accent-track grit.
        for id in SoundId::MOVEMENT_MOTIFS {
            assert!(!is_glitch_family(id), "{}", id.name());
        }
    }

    #[test]
    fn grammar_cues_are_immediate_except_the_first_light_bloom() {
        for id in SoundId::GRAMMAR {
            let want = if id == SoundId::FirstLight {
                Quantize::Beat(BeatClass::Strong)
            } else {
                Quantize::Immediate
            };
            assert_eq!(quantize_of(id), want, "{}", id.name());
        }
        // Only the cycle refusal takes the harsh accent path.
        for id in SoundId::GRAMMAR {
            assert_eq!(
                is_glitch_family(id),
                id == SoundId::RefuseCycle,
                "{}",
                id.name()
            );
        }
    }

    #[test]
    fn stinger_track_routing() {
        // Only the corruption tick takes the harsh accent path; the other two
        // stingers sit in the master fog with the rest of the tonal material.
        assert!(is_glitch_family(SoundId::StingerCorrupt));
        assert!(!is_glitch_family(SoundId::StingerNotice));
        assert!(!is_glitch_family(SoundId::StingerMemory));
    }
}
