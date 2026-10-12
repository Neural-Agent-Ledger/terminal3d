//! `ambience` — the per-level / per-place mood abstraction. **Pure logic, no device.**
//!
//! The four "dying-machine" drone beds ([`crate::synth::SoundId::DRONE_MOODS`])
//! are exposed to the game as named *moods* rather than raw sound ids, plus a
//! single `intensity: f32` knob in `[0, 1]` that scales the bed's *presence*:
//! its volume, its filter openness, and its reverb/movement depth. The game
//! sends an [`SetAmbience`](crate::plugin::SetAmbience) message **per level**
//! (on load) and **per place/zone** (on a trigger); when the mood or intensity
//! changes at runtime the device layer crossfades smoothly from the current bed
//! to the new one (a timed gain ramp on kira, never a hard cut).
//!
//! This module owns the engine-agnostic decisions:
//! * [`AmbienceMood`] — the named moods and their drone-bed mapping
//!   ([`AmbienceMood::drone`]).
//! * [`Ambience`] — a `{ mood, intensity }` request.
//! * [`AmbienceTargets`] — the *target* device params an intensity implies
//!   (volume, filter cutoff, reverb mix, vibrato/movement depth). This is the
//!   single place the "intensity → dread" curve lives; `playback` only applies
//!   the numbers, it does not decide them. That keeps the horror-progression
//!   curve unit-testable with no audio device.
//! * [`Crossfade`] — a pure, time-driven two-bed crossfade state machine. Given
//!   the elapsed fraction of a fade it returns the outgoing/incoming gains;
//!   `playback` feeds it real wall-clock time. Testable with no device.
//!
//! Determinism: nothing here renders PCM or touches a device, so it cannot
//! perturb the synth's golden hashes. The drone PCM is unchanged; ambience only
//! *selects* a bed and *scales* device-side mixer params.

use crate::synth::SoundId;

/// The intensity a legacy [`SetAmbient`](crate::plugin::SetAmbient) call (or any
/// caller that doesn't specify one) gets: a moderate, mid-dread presence.
pub const DEFAULT_INTENSITY: f32 = 0.5;

/// A named atmospheric mood. Each maps 1:1 to one of the four evolving drone
/// beds. The game picks a mood per level and (optionally) overrides it per
/// place/zone; the device layer crossfades when it changes.
///
/// Story tone (a host game's narrative maps its moods onto these):
/// * [`ColdVoid`](AmbienceMood::ColdVoid) — empty, frozen, abandoned space.
/// * [`ServerHum`](AmbienceMood::ServerHum) — the live machine still running,
///   electrical and indifferent.
/// * [`OrganicDecay`](AmbienceMood::OrganicDecay) — wet, rotting, biological
///   failure of a once-clean system.
/// * [`AlarmTension`](AmbienceMood::AlarmTension) — imminent threat; the
///   `while(true)` closing in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AmbienceMood {
    /// Cold, empty void. Drone bed: [`SoundId::DroneColdVoid`].
    ColdVoid,
    /// Electrical server hum. Drone bed: [`SoundId::DroneServerHum`].
    ServerHum,
    /// Wet organic decay. Drone bed: [`SoundId::DroneOrganicDecay`].
    OrganicDecay,
    /// Alarm / imminent-threat tension. Drone bed: [`SoundId::DroneAlarmTension`].
    AlarmTension,
}

impl AmbienceMood {
    /// Every mood, in canonical order (matches [`SoundId::DRONE_MOODS`]).
    pub const ALL: [AmbienceMood; 4] = [
        AmbienceMood::ColdVoid,
        AmbienceMood::ServerHum,
        AmbienceMood::OrganicDecay,
        AmbienceMood::AlarmTension,
    ];

    /// The drone-bed [`SoundId`] this mood plays. This is the load-bearing
    /// mood → bed mapping the whole ambience system routes through.
    #[inline]
    pub fn drone(self) -> SoundId {
        match self {
            AmbienceMood::ColdVoid => SoundId::DroneColdVoid,
            AmbienceMood::ServerHum => SoundId::DroneServerHum,
            AmbienceMood::OrganicDecay => SoundId::DroneOrganicDecay,
            AmbienceMood::AlarmTension => SoundId::DroneAlarmTension,
        }
    }

    /// The inverse mapping: the mood whose bed is `id`, if any. Lets the game
    /// translate a raw drone [`SoundId`] back into a mood (e.g. when migrating a
    /// legacy [`SetAmbient`](crate::plugin::SetAmbient) call).
    #[inline]
    pub fn from_drone(id: SoundId) -> Option<AmbienceMood> {
        AmbienceMood::ALL.into_iter().find(|m| m.drone() == id)
    }

    /// A short stable name for logs/tests.
    pub fn name(self) -> &'static str {
        match self {
            AmbienceMood::ColdVoid => "ColdVoid",
            AmbienceMood::ServerHum => "ServerHum",
            AmbienceMood::OrganicDecay => "OrganicDecay",
            AmbienceMood::AlarmTension => "AlarmTension",
        }
    }
}

/// A full ambience request: which mood, and how present it is.
///
/// `intensity` is clamped to `[0, 1]` on construction. The game raises it as
/// corruption / the `while(true)` closes in — higher intensity = more dread
/// (louder, brighter filter, deeper movement). See [`AmbienceTargets`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ambience {
    /// The selected mood (drone bed).
    pub mood: AmbienceMood,
    /// Presence in `[0, 1]`. 0 = barely-there bed; 1 = full dread.
    pub intensity: f32,
}

impl Ambience {
    /// Build a request, clamping `intensity` into `[0, 1]`.
    #[inline]
    pub fn new(mood: AmbienceMood, intensity: f32) -> Ambience {
        Ambience {
            mood,
            intensity: intensity.clamp(0.0, 1.0),
        }
    }

    /// The drone bed this request plays.
    #[inline]
    pub fn drone(self) -> SoundId {
        self.mood.drone()
    }

    /// The device-side target params this request implies.
    #[inline]
    pub fn targets(self) -> AmbienceTargets {
        AmbienceTargets::for_intensity(self.intensity)
    }
}

/// The device-side target parameters an [`Ambience`] intensity implies.
///
/// This is the single authority for the **intensity → dread** curve. `playback`
/// reads these numbers and applies them to the live bed (tweened); it does not
/// re-derive them. Keeping the curve here (not in the device module) makes the
/// horror progression a pure, unit-testable function.
///
/// All fields rise monotonically with intensity, so driving intensity up always
/// increases dread: louder, brighter (filter opens), wetter (more reverb), and
/// more alive (deeper movement / vibrato on the bed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbienceTargets {
    /// Linear bed gain in `[0, 1]`. At intensity 0 the bed is quiet but audible
    /// ([`Self::MIN_GAIN`]); at 1 it is at full presence.
    pub gain: f32,
    /// Lowpass cutoff in Hz. Low intensity = muffled/distant; high = open/present.
    pub filter_cutoff_hz: f32,
    /// Reverb wet mix in `[0, 1]`. Higher intensity = deeper, more enveloping space.
    pub reverb_mix: f32,
    /// Movement depth in `[0, 1]`: how much the bed's slow LFO / detune sways.
    /// Higher intensity = a more "alive", unstable, dread-laden bed. (Advisory;
    /// applied via reverb/filter sway since the synth LFO is baked into the PCM.)
    pub movement: f32,
}

impl AmbienceTargets {
    /// Bed gain floor: even at intensity 0 the bed is present, not silent — a
    /// dying machine is never fully quiet.
    pub const MIN_GAIN: f32 = 0.25;
    /// Filter cutoff at intensity 0 (Hz): muffled, distant.
    pub const MIN_CUTOFF_HZ: f32 = 900.0;
    /// Filter cutoff at intensity 1 (Hz): open, present, in-the-room.
    pub const MAX_CUTOFF_HZ: f32 = 6000.0;
    /// Reverb wet mix at intensity 0.
    pub const MIN_REVERB_MIX: f32 = 0.18;
    /// Reverb wet mix at intensity 1.
    pub const MAX_REVERB_MIX: f32 = 0.55;

    /// Map an intensity in `[0, 1]` to its target device params. The mapping is
    /// monotonic in every field, so intensity is a pure dread dial.
    pub fn for_intensity(intensity: f32) -> AmbienceTargets {
        let t = intensity.clamp(0.0, 1.0);
        AmbienceTargets {
            gain: lerp(Self::MIN_GAIN, 1.0, t),
            filter_cutoff_hz: lerp(Self::MIN_CUTOFF_HZ, Self::MAX_CUTOFF_HZ, t),
            reverb_mix: lerp(Self::MIN_REVERB_MIX, Self::MAX_REVERB_MIX, t),
            movement: t,
        }
    }
}

/// Linear interpolation; `t` is assumed already clamped to `[0, 1]`.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// A pure, time-driven crossfade between an *outgoing* and an *incoming* bed.
///
/// When the active mood changes, the device layer starts a [`Crossfade`] of a
/// configured `duration` and, each frame, asks it for the outgoing/incoming
/// gains at the current elapsed time. The fade is an equal-power-ish ramp so
/// the perceived loudness stays roughly constant through the transition (no dip,
/// no hard cut). When `elapsed >= duration` the fade is [`Crossfade::done`] and
/// the outgoing bed can be stopped.
///
/// Pure: it holds only the duration and the moods; the caller supplies elapsed
/// time. No device, no kira types — unit-testable in full.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossfade {
    /// The mood fading out (the previously-active bed).
    pub from: AmbienceMood,
    /// The mood fading in (the newly-requested bed).
    pub to: AmbienceMood,
    /// Total fade duration in seconds (`> 0`).
    pub duration_secs: f32,
}

impl Crossfade {
    /// Start a crossfade from `from` to `to` over `duration_secs` (clamped to a
    /// sane minimum so a zero/negative duration can't divide-by-zero).
    pub fn new(from: AmbienceMood, to: AmbienceMood, duration_secs: f32) -> Crossfade {
        Crossfade {
            from,
            to,
            duration_secs: duration_secs.max(Self::MIN_DURATION),
        }
    }

    /// Smallest fade duration (seconds) — avoids a hard cut even if the caller
    /// asks for 0.
    pub const MIN_DURATION: f32 = 0.05;

    /// The normalized progress `[0, 1]` of the fade at `elapsed_secs`.
    #[inline]
    pub fn progress(self, elapsed_secs: f32) -> f32 {
        (elapsed_secs / self.duration_secs).clamp(0.0, 1.0)
    }

    /// `true` once the fade has fully completed at `elapsed_secs`.
    #[inline]
    pub fn done(self, elapsed_secs: f32) -> bool {
        elapsed_secs >= self.duration_secs
    }

    /// The `(outgoing_gain, incoming_gain)` envelope at `elapsed_secs`, each in
    /// `[0, 1]`. Equal-power: `out = cos(p·π/2)`, `in = sin(p·π/2)`, so the two
    /// gains' squares always sum to 1 (constant perceived power — no mid-fade
    /// dip, no doubling). The caller multiplies these by each bed's
    /// intensity-driven target gain.
    #[inline]
    pub fn gains(self, elapsed_secs: f32) -> (f32, f32) {
        let p = self.progress(elapsed_secs);
        let angle = p * std::f32::consts::FRAC_PI_2;
        (angle.cos(), angle.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mood_maps_to_a_distinct_drone_bed() {
        let mut seen = std::collections::HashSet::new();
        for m in AmbienceMood::ALL {
            assert!(seen.insert(m.drone()), "{} reused a drone bed", m.name());
        }
        // And the mapping matches the synth's canonical drone-mood order.
        for (m, &id) in AmbienceMood::ALL.iter().zip(SoundId::DRONE_MOODS.iter()) {
            assert_eq!(m.drone(), id, "{} mapped to the wrong bed", m.name());
        }
    }

    #[test]
    fn mood_round_trips_through_drone_id() {
        for m in AmbienceMood::ALL {
            assert_eq!(AmbienceMood::from_drone(m.drone()), Some(m));
        }
        // A non-drone sound has no mood.
        assert_eq!(AmbienceMood::from_drone(SoundId::Boot), None);
    }

    #[test]
    fn ambience_clamps_intensity() {
        assert_eq!(Ambience::new(AmbienceMood::ColdVoid, 5.0).intensity, 1.0);
        assert_eq!(Ambience::new(AmbienceMood::ColdVoid, -3.0).intensity, 0.0);
    }

    #[test]
    fn intensity_is_a_monotonic_dread_dial() {
        // Every target param rises (or holds) as intensity climbs: more dread.
        let lo = AmbienceTargets::for_intensity(0.0);
        let mid = AmbienceTargets::for_intensity(0.5);
        let hi = AmbienceTargets::for_intensity(1.0);

        assert!(lo.gain < mid.gain && mid.gain < hi.gain, "gain not rising");
        assert!(
            lo.filter_cutoff_hz < mid.filter_cutoff_hz
                && mid.filter_cutoff_hz < hi.filter_cutoff_hz,
            "cutoff not opening"
        );
        assert!(
            lo.reverb_mix < mid.reverb_mix && mid.reverb_mix < hi.reverb_mix,
            "reverb not deepening"
        );
        assert!(lo.movement < hi.movement, "movement not deepening");

        // Endpoints hit the declared bounds.
        assert_eq!(lo.gain, AmbienceTargets::MIN_GAIN);
        assert_eq!(hi.gain, 1.0);
        assert_eq!(lo.filter_cutoff_hz, AmbienceTargets::MIN_CUTOFF_HZ);
        assert_eq!(hi.filter_cutoff_hz, AmbienceTargets::MAX_CUTOFF_HZ);
    }

    #[test]
    fn never_fully_silent_at_zero_intensity() {
        // A dying machine is never truly quiet — the bed floor is audible.
        let t = AmbienceTargets::for_intensity(0.0);
        assert!(t.gain >= AmbienceTargets::MIN_GAIN);
        assert!(t.gain > 0.0);
    }

    #[test]
    fn crossfade_is_a_ramp_not_a_hard_cut() {
        let cf = Crossfade::new(AmbienceMood::ColdVoid, AmbienceMood::AlarmTension, 2.0);

        // Start: fully outgoing, no incoming.
        let (out0, in0) = cf.gains(0.0);
        assert!((out0 - 1.0).abs() < 1e-5 && in0.abs() < 1e-5);
        assert!(!cf.done(0.0));

        // Midpoint: BOTH beds audible at once — overlap, the proof it's not a cut.
        let (outm, inm) = cf.gains(1.0);
        assert!(
            outm > 0.1 && inm > 0.1,
            "no overlap → hard cut, not a crossfade"
        );

        // End: fully incoming, outgoing gone, and marked done.
        let (oute, ine) = cf.gains(2.0);
        assert!(oute.abs() < 1e-5 && (ine - 1.0).abs() < 1e-5);
        assert!(cf.done(2.0));
    }

    #[test]
    fn crossfade_is_equal_power() {
        // out² + in² == 1 across the fade → constant perceived loudness.
        let cf = Crossfade::new(AmbienceMood::ServerHum, AmbienceMood::OrganicDecay, 1.0);
        for step in 0..=10 {
            let e = step as f32 / 10.0;
            let (o, i) = cf.gains(e);
            assert!((o * o + i * i - 1.0).abs() < 1e-4, "not equal-power at {e}");
        }
    }

    #[test]
    fn crossfade_clamps_zero_duration() {
        // A 0-second request must not produce a divide-by-zero / NaN — it floors
        // to a tiny ramp, still overlapping for an instant.
        let cf = Crossfade::new(AmbienceMood::ColdVoid, AmbienceMood::ServerHum, 0.0);
        assert!(cf.duration_secs >= Crossfade::MIN_DURATION);
        let (o, i) = cf.gains(0.0);
        assert!(o.is_finite() && i.is_finite());
    }
}
