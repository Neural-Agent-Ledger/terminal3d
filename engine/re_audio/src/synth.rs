//! `synth` — in-house procedural audio synthesis. **Pure Rust maths.**
//!
//! A 1:1 port of the legacy `src/synth.c` / `synth.h`. Renders TERMINAL STATE's ~14
//! sound effects + ambient drones entirely from oscillators, an ADSR envelope,
//! and deterministic noise — there are *no* `.wav` / `.ogg` assets on disk.
//! Each sound is produced as an in-memory mono PCM buffer at 44 100 Hz.
//!
//! This module is the analogue of the raycaster being pure compute: it knows
//! nothing about an audio device. The [`crate::playback`] layer is the output
//! side (rodio), exactly as miniaudio was in the C build.
//!
//! **Determinism:** noise uses a fixed-seed LCG (never `rand`/clock), so the
//! same recipe renders identical PCM on every run of a given build. This is
//! determinism-tested (build-vs-build); it is *not* pinned to a committed
//! golden baseline.

use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Sample rate for every rendered sound (matches the legacy engine rate).
pub const SAMPLE_RATE: u32 = 44_100;

const TWO_PI: f32 = 2.0 * PI;

// ───────────────────────────── waveforms ────────────────────────────────────

/// Oscillator waveforms. [`Wave::Noise`] produces deterministic filtered noise
/// (see [`noise`]) rather than a periodic shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    /// Pure sine — the softest, most tonal partial shape.
    Sine,
    /// Hard `+1`/`-1` square — bright, buzzy, rich in odd harmonics.
    Square,
    /// Linear triangle — softer than a square, brighter than a sine.
    Tri,
    /// Linear ramp — bright, rich in all harmonics; used for grit and edge.
    Saw,
    /// Deterministic filtered noise via [`noise`] rather than a periodic shape.
    Noise,
}

/// Oscillator sample for a given waveform at `phase01` in `[0, 1)`.
/// Output is in `[-1, 1]`. [`Wave::Noise`] is not periodic and returns `0.0`
/// here (callers use [`noise`] instead), matching the legacy `synth_osc`.
#[inline]
pub fn osc(wave: Wave, mut phase01: f32) -> f32 {
    // Wrap defensively so callers can pass accumulated phase.
    phase01 -= phase01.floor();
    if phase01 < 0.0 {
        phase01 += 1.0;
    }

    match wave {
        Wave::Sine => (TWO_PI * phase01).sin(),
        Wave::Square => {
            if phase01 < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        // Conventional triangle: 0 at phase 0, +1 at 1/4, 0 at 1/2,
        // -1 at 3/4, back to 0 at 1.
        Wave::Tri => {
            if phase01 < 0.25 {
                4.0 * phase01
            } else if phase01 < 0.75 {
                2.0 - 4.0 * phase01
            } else {
                4.0 * phase01 - 4.0
            }
        }
        Wave::Saw => 2.0 * phase01 - 1.0,
        Wave::Noise => 0.0,
    }
}

// ───────────────────────── deterministic noise ──────────────────────────────

/// 32-bit LCG state (Numerical Recipes constants). The caller owns the state,
/// so noise is reproducible: same seed → same sequence on every platform.
#[derive(Clone, Copy, Debug)]
pub struct Lcg(pub u32);

impl Lcg {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
}

/// Deterministic white-noise sample in `[-1, 1)` from the LCG, advanced in
/// place. Mirrors the legacy `synth_noise` (top 24 bits → `[0, 1)` → `[-1, 1)`).
#[inline]
pub fn noise(lcg: &mut Lcg) -> f32 {
    let r = lcg.next_u32();
    let u = (r >> 8) as f32 / 16_777_216.0; // 2^24
    u * 2.0 - 1.0
}

// ───────────────────────────── envelope ─────────────────────────────────────

/// ADSR amplitude envelope, all times in seconds. `sustain` is a level in
/// `[0, 1]`, not a time; the sustain segment fills whatever duration remains
/// between `(attack + decay)` and `(total - release)`.
#[derive(Clone, Copy, Debug)]
pub struct Envelope {
    /// Time to rise from silence to full amplitude, in seconds.
    pub attack: f32,
    /// Time to fall from full amplitude down to the `sustain` level, in seconds.
    pub decay: f32,
    /// Level held during the sustain segment, in `[0, 1]` (a level, not a duration).
    pub sustain: f32,
    /// Time to fall from the sustain level to silence, in seconds.
    pub release: f32,
}

/// ADSR amplitude at sample index `i` of a `total_frames`-long sound. Returns a
/// gain in `[0, 1]`. Monotonic non-decreasing across attack, non-increasing
/// across release. A faithful port of the legacy `synth_env` (computed in
/// `f64` then narrowed, so output is bit-stable vs. the C build).
pub fn env(e: &Envelope, i: usize, total_frames: usize, sample_rate: u32) -> f32 {
    if total_frames == 0 {
        return 0.0;
    }

    let t = i as f64 / sample_rate as f64;
    let total_s = total_frames as f64 / sample_rate as f64;

    let a = if e.attack < 0.0 { 0.0 } else { e.attack as f64 };
    let d = if e.decay < 0.0 { 0.0 } else { e.decay as f64 };
    let r = if e.release < 0.0 {
        0.0
    } else {
        e.release as f64
    };
    let s = (e.sustain as f64).clamp(0.0, 1.0);

    let rel_start = (total_s - r).max(0.0);

    let mut g = if t < a && a > 0.0 {
        // Attack: linear 0 → 1.
        t / a
    } else if t < a + d && d > 0.0 {
        // Decay: linear 1 → sustain.
        let dt = (t - a) / d;
        1.0 - dt * (1.0 - s)
    } else if t < rel_start {
        // Sustain hold.
        s
    } else {
        // Release: linear (level at rel_start) → 0.
        let level = if rel_start <= a {
            1.0
        } else if rel_start <= a + d && d > 0.0 {
            1.0 - ((rel_start - a) / d) * (1.0 - s)
        } else {
            s
        };
        if r > 0.0 {
            let rt = ((t - rel_start) / r).min(1.0);
            level * (1.0 - rt)
        } else {
            0.0
        }
    };

    g = g.clamp(0.0, 1.0);
    g as f32
}

// ──────────────────────────── recipe model ──────────────────────────────────
//
// Each sound is built from up to N partials (oscillator + relative gain +
// optional linear frequency glide + optional vibrato) plus an optional noise
// mix, shaped by an ADSR envelope and a master gain. The table keeps the 14
// recipes declarative and side-effect-free — same structure as legacy `Recipe`.

const MAX_PARTIALS: usize = 4;

#[derive(Clone, Copy)]
struct Partial {
    wave: Wave,
    freq0: f32,         // Hz at t = 0
    freq1: f32,         // Hz at t = duration (glide); == freq0 for no glide
    gain: f32,          // relative amplitude
    vibrato_hz: f32,    // 0 = none; else freq-wobble rate
    vibrato_depth: f32, // fractional vibrato depth (e.g. 0.01 = 1 %)
}

const fn p(
    wave: Wave,
    freq0: f32,
    freq1: f32,
    gain: f32,
    vibrato_hz: f32,
    vibrato_depth: f32,
) -> Partial {
    Partial {
        wave,
        freq0,
        freq1,
        gain,
        vibrato_hz,
        vibrato_depth,
    }
}

struct Recipe {
    name: &'static str,
    duration_s: f32,
    master_gain: f32, // overall level, leaves headroom (< 1)
    env: Envelope,
    noise_mix: f32, // 0..1 amount of noise blended in
    noise_lp: f32,  // one-pole low-pass coeff for noise [0..1); 0 = white
    partial_count: usize,
    partials: [Partial; MAX_PARTIALS],
    seed: u32, // fixed LCG seed for this sound's noise
}

/// Identifiers for the synthesised sounds, in [`SynthBank`] field order.
/// Indices 0..=7 are the originally-registered sounds; 8..=13 are additive;
/// 14..=27 are the rhythmic-audio additions (glitch family, drone layers, and
/// the minor-pentatonic tonal palette); 28..=31 the atmosphere drone moods;
/// 32..=34 the narrative stingers; 35..=38 the campaign-movement motifs.
/// **The order is load-bearing for the determinism tests — only ever APPEND,
/// never reorder existing entries.**
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SoundId {
    /// Rising two-tone chime on game boot, ~1.6 s, bright and clean.
    Boot = 0,
    /// Tiny percussive tick for a pipe-switch toggle, ~0.06 s.
    SwitchClick = 1,
    /// Soft confirm blip for a generic interact, ~0.18 s — rounder than the switch.
    Interact = 2,
    /// Low, slow mechanical swell for a door opening, ~2.6 s.
    Door = 3,
    /// Harsh digital burst, ~0.35 s — mostly noise plus a buzzy saw.
    Glitch = 4,
    /// Wordless tonal "thought voice", ~3.0 s, vowel-ish with vibrato.
    Voice = 5,
    /// Soft flowing whoosh for a pipe, ~2.0 s, dominated by filtered noise.
    Pipe = 6,
    /// Evolving low drone, ~10 s, designed to loop seamlessly.
    Ambient = 7,
    /// Short soft thud for a footstep, ~0.12 s — low sine body plus a quick noise burst.
    Footstep = 8,
    /// Brief bright blip for menu navigation, ~0.05 s.
    UiMove = 9,
    /// Confirm tone for a menu selection: longer and rounder than [`SoundId::UiMove`],
    /// rises a step, ~0.16 s — reads as a positive "accepted" cue.
    UiSelect = 10,
    /// Short rising arpeggio/chord for solving a level, ~0.9 s.
    WinSting = 11,
    /// Slow dark drone for the "cold" level theme, ~12 s.
    AmbientCold = 12,
    /// Warm gentle hum for the "garden" level theme, ~12 s.
    AmbientGarden = 13,
    // ── rhythmic-audio additions (appended; never reordered) ──
    /// Micro bit-crush click, ~0.03 s — a tiny high square transient with a
    /// touch of white noise: the "data tick" between beats. Routed to the
    /// accent track as part of the terminal-glitch family.
    GlitchTick = 14,
    /// Short filtered-noise burst, ~0.16 s, like a packet of corrupted data.
    /// Terminal-glitch family (accent track).
    DataBurst = 15,
    /// Fast descending frequency sweep, ~0.22 s — a CRT scanline whine
    /// dropping through the spectrum. Terminal-glitch family (accent track).
    Scanline = 16,
    /// Short retriggered buzz, ~0.18 s — a square gated by a fast wobble to
    /// read as a "stutter". Terminal-glitch family (accent track).
    Stutter = 17,
    /// Slow evolving mid drone, ~14 s, layered under the ambient bed to
    /// thicken it.
    DroneLayerA = 18,
    /// Higher, airier drone than [`SoundId::DroneLayerA`], ~14 s, with an
    /// even slower LFO.
    DroneLayerB = 19,
    // Minor-pentatonic tonal palette (A minor pentatonic, ~1.5 octaves),
    // pre-rendered so the director can pick pitched notes deterministically.
    /// Tonal-palette note A3 (220.00 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneA3 = 20,
    /// Tonal-palette note C4 (261.63 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneC4 = 21,
    /// Tonal-palette note D4 (293.66 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneD4 = 22,
    /// Tonal-palette note E4 (329.63 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneE4 = 23,
    /// Tonal-palette note G4 (392.00 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneG4 = 24,
    /// Tonal-palette note A4 (440.00 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneA4 = 25,
    /// Tonal-palette note C5 (523.25 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneC5 = 26,
    /// Tonal-palette note D5 (587.33 Hz), short plucked/bell-ish note, ~0.7 s.
    ToneD5 = 27,
    // ── atmosphere drone moods (appended; never reordered) ──
    // Four evolving "dying-machine" drone beds. Each is a long, seamless-looping
    // sustained tone with detuned/beating partials, a slow LFO (vibrato) and a
    // filtered-noise floor. `playback` routes these through the reverb/lowpass
    // fog track and crossfades between them; the game selects one per level/place
    // via [`crate::ambience`].
    /// Deepest and darkest drone mood: an empty buffer, the void between dead
    /// processes. Sub-bass root with a near-unison detune (~0.6 Hz beat), a
    /// hollow fifth, and a heavy dark noise floor. ~16 s.
    DroneColdVoid = 28,
    /// Dying-machine drone mood: a mains-style hum (~50 Hz) plus its
    /// harmonics, slightly unstable, with mid-bright filtered noise reading
    /// as fan/disk air. ~15 s.
    DroneServerHum = 29,
    /// Warmer, breathing drone mood for rot/garden spaces: a low minor triad
    /// with a wider, slower LFO so it swells like something alive but
    /// failing, over breathier mid noise. ~16 s.
    DroneOrganicDecay = 30,
    /// Rising-dread drone mood: a beating minor-second cluster plus a higher
    /// unstable partial with no resolution, a faster/shallower LFO, and a
    /// brighter noise edge. ~14 s.
    DroneAlarmTension = 31,
    // ── narrative trigger stingers (appended; never reordered) ──
    // Short (< 1.5 s) punctuation for story `[trigger]` fires. Cold terminal
    // horror, MNEMOS/LETHE ambient-pull: no fanfare, no resolved major chord.
    /// Cold "system notice" ping — a detuned high sine pair with a fast decay.
    /// Reads as the facility acknowledging you, not congratulating you.
    /// [`crate::policy::quantize_of`] → `Immediate` (it is a reaction).
    StingerNotice = 32,
    /// "Memory surfacing" swell — a dark filtered-noise bed out of which a
    /// beating detuned pair glides into near-unison focus. Long attack, no
    /// transient. Quantized to `Beat(Strong)` so it lands on a bar downbeat.
    StingerMemory = 33,
    /// "Corruption tick" — a filtered, wobble-gated downward glitch transient.
    /// Part of the glitch *routing* family (accent track: distortion + delay)
    /// but deliberately NOT in [`SoundId::GLITCH_FAMILY`], so the generative
    /// director never fires it as ambient texture: it only means "story".
    StingerCorrupt = 34,
    // ── movement motifs (appended; never reordered) ──
    // Textures for the four campaign-movement themes (`Theme::P1..P4`). Each is
    // a *texture*, never a lead: MNEMOS/LETHE stay ambient forces, so nothing
    // here is voice-like, melodic or fanfare-shaped. The generative
    // [`crate::director`] fires them as the movement's motif; they are listed in
    // [`SoundId::MOVEMENT_MOTIFS`] and, like the stingers, are deliberately
    // absent from `GLITCH_FAMILY` / `TONAL_PALETTE`.
    /// `P2TheGaze` motif — a soft, slightly detuned low-mid pulse (two sines a
    /// couple of Hz apart, so it *beats* rather than sits still). No transient:
    /// it swells under the bed like something breathing while it watches.
    GazePulse = 35,
    /// `P3Grief` motif — the LETHE river wash: a very dark filtered-noise
    /// swell with a long decay over a faint low tone. Hollow, no pitch centre
    /// to hold onto.
    RiverWash = 36,
    /// `P4Closing` motif — the first heartbeat thump ("lub"): a low soft body
    /// with a fast attack, landing on the bar downbeat.
    HeartLub = 37,
    /// `P4Closing` motif — the answering thump ("dub"): lower and quieter than
    /// [`SoundId::HeartLub`], fired on the beat right after it.
    HeartDub = 38,
    // ── the puzzle SOUND GRAMMAR (appended; never reordered) ──
    // Every cue below answers exactly one game event, and all of them speak the
    // same four-word grammar (see `crate::cue` for the event → sound table):
    //   * rising pitch   = progress / something opened
    //   * falling pitch  = regress / something closed / refusal
    //   * consonant      = correct (simple partial ratios: 1, 3/2, 2, 5/4)
    //   * dissonant      = blocked / over budget (minor-second beating pairs)
    /// A pipe switched ON: a click whose tone rises a perfect fifth (A4 → E5).
    SwitchOn = 39,
    /// A pipe switched OFF: the mirror of [`SoundId::SwitchOn`] — the tone falls
    /// a fifth and sits darker and quieter. ON and OFF never sound alike.
    SwitchOff = 40,
    /// Interact pressed on nothing: a dull, pitch-neutral thud. Deliberately not
    /// a rising blip — a press that did nothing must not sound like success.
    InteractEmpty = 41,
    /// Refused: the power budget is spent. A "brownout" — two saws a minor
    /// second apart (beating, dissonant) sagging downward.
    RefusePower = 42,
    /// Refused: the toggle would close a forbidden loop. A stuttering,
    /// semitone-clashing fall (accent track).
    RefuseCycle = 43,
    /// A node started firing: a glassy chime that scoops UP into its note. The
    /// cue layer pitches it higher the more of the network is lit.
    NodeLit = 44,
    /// A node went dark: the chime's falling, minor-second-shadowed inverse.
    NodeUnlit = 45,
    /// The network is satisfied (the door will open): a consonant major bloom.
    CircuitResolve = 46,
    /// The crosshair just landed on something interactable: a tiny, quiet
    /// rising blip ("there is something here").
    AimTick = 47,
    /// A wall placard / note came into reading range: a soft page-open swish.
    NoteRead = 48,
    /// The player crossed a non-euclidean portal: a space-bending rising sweep.
    PortalWarp = 49,
    /// Climbed to a higher storey: a low rising swell.
    StoreyUp = 50,
    /// Descended to a lower storey: the same swell, falling.
    StoreyDown = 51,
    /// Footstep on metal (grates, gold, neon, monitor floors): a short clang.
    FootstepMetal = 52,
    /// Footstep on wet ground (wet stone, decay, lava crust): a gloop.
    FootstepWet = 53,
    /// Footstep on soft ground (snow, frost, organic): a muffled crunch.
    FootstepSoft = 54,
    /// A colour shard was recovered ("First Light"): a slow consonant bloom.
    FirstLight = 55,
}

impl SoundId {
    /// Every sound id, in bank order.
    pub const ALL: [SoundId; SYNTH_SOUND_COUNT] = [
        SoundId::Boot,
        SoundId::SwitchClick,
        SoundId::Interact,
        SoundId::Door,
        SoundId::Glitch,
        SoundId::Voice,
        SoundId::Pipe,
        SoundId::Ambient,
        SoundId::Footstep,
        SoundId::UiMove,
        SoundId::UiSelect,
        SoundId::WinSting,
        SoundId::AmbientCold,
        SoundId::AmbientGarden,
        SoundId::GlitchTick,
        SoundId::DataBurst,
        SoundId::Scanline,
        SoundId::Stutter,
        SoundId::DroneLayerA,
        SoundId::DroneLayerB,
        SoundId::ToneA3,
        SoundId::ToneC4,
        SoundId::ToneD4,
        SoundId::ToneE4,
        SoundId::ToneG4,
        SoundId::ToneA4,
        SoundId::ToneC5,
        SoundId::ToneD5,
        SoundId::DroneColdVoid,
        SoundId::DroneServerHum,
        SoundId::DroneOrganicDecay,
        SoundId::DroneAlarmTension,
        SoundId::StingerNotice,
        SoundId::StingerMemory,
        SoundId::StingerCorrupt,
        SoundId::GazePulse,
        SoundId::RiverWash,
        SoundId::HeartLub,
        SoundId::HeartDub,
        SoundId::SwitchOn,
        SoundId::SwitchOff,
        SoundId::InteractEmpty,
        SoundId::RefusePower,
        SoundId::RefuseCycle,
        SoundId::NodeLit,
        SoundId::NodeUnlit,
        SoundId::CircuitResolve,
        SoundId::AimTick,
        SoundId::NoteRead,
        SoundId::PortalWarp,
        SoundId::StoreyUp,
        SoundId::StoreyDown,
        SoundId::FootstepMetal,
        SoundId::FootstepWet,
        SoundId::FootstepSoft,
        SoundId::FirstLight,
    ];

    /// The puzzle sound-grammar cues (indices 39..), in enum order. Like the
    /// stingers they are *meaning*, never ambient texture: the generative
    /// director must never pick one (see the palette-separation test).
    pub const GRAMMAR: [SoundId; 17] = [
        SoundId::SwitchOn,
        SoundId::SwitchOff,
        SoundId::InteractEmpty,
        SoundId::RefusePower,
        SoundId::RefuseCycle,
        SoundId::NodeLit,
        SoundId::NodeUnlit,
        SoundId::CircuitResolve,
        SoundId::AimTick,
        SoundId::NoteRead,
        SoundId::PortalWarp,
        SoundId::StoreyUp,
        SoundId::StoreyDown,
        SoundId::FootstepMetal,
        SoundId::FootstepWet,
        SoundId::FootstepSoft,
        SoundId::FirstLight,
    ];

    /// The movement motifs, in `Theme::P1..P4` order of use: the gaze pulse
    /// (`P2`), the LETHE river wash (`P3`) and the heartbeat pair (`P4`).
    /// `P1FirstLight` has no motif — its character is the *absence* of one (a
    /// warm, sparse pentatonic and almost no glitch).
    ///
    /// These are theme textures, not director ambient texture and not story
    /// punctuation: like [`SoundId::STINGERS`] they stay out of
    /// [`SoundId::GLITCH_FAMILY`] and [`SoundId::TONAL_PALETTE`].
    pub const MOVEMENT_MOTIFS: [SoundId; 4] = [
        SoundId::GazePulse,
        SoundId::RiverWash,
        SoundId::HeartLub,
        SoundId::HeartDub,
    ];

    /// The narrative trigger stingers, in canonical order. Each is a short
    /// (< 1.5 s) punctuation cue fired by a story `[trigger]`, never by the
    /// generative [`crate::director`]. The game maps a trigger label → one of
    /// these via the host game's own trigger table.
    pub const STINGERS: [SoundId; 3] = [
        SoundId::StingerNotice,
        SoundId::StingerMemory,
        SoundId::StingerCorrupt,
    ];

    /// The four atmosphere drone-mood beds, in [`crate::ambience::AmbienceMood`]
    /// order. Each is a long, seamless-looping evolving drone routed through the
    /// reverb/lowpass fog track and crossfaded by `playback`.
    pub const DRONE_MOODS: [SoundId; 4] = [
        SoundId::DroneColdVoid,
        SoundId::DroneServerHum,
        SoundId::DroneOrganicDecay,
        SoundId::DroneAlarmTension,
    ];

    /// The minor-pentatonic tonal palette, low → high (A minor pentatonic).
    /// The director picks from this for sparse pitched notes on strong beats.
    pub const TONAL_PALETTE: [SoundId; 8] = [
        SoundId::ToneA3,
        SoundId::ToneC4,
        SoundId::ToneD4,
        SoundId::ToneE4,
        SoundId::ToneG4,
        SoundId::ToneA4,
        SoundId::ToneC5,
        SoundId::ToneD5,
    ];

    /// The terminal-glitch family. The director picks from this for busy weak
    /// beats; `playback` routes these to the harsher accent track.
    pub const GLITCH_FAMILY: [SoundId; 4] = [
        SoundId::GlitchTick,
        SoundId::DataBurst,
        SoundId::Scanline,
        SoundId::Stutter,
    ];

    /// The recipe table index for this sound.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// The recipe's human name (matches the legacy MANIFEST names).
    pub fn name(self) -> &'static str {
        RECIPES[self.index()].name
    }

    /// Resolve a recipe [`name`](SoundId::name) back to its id, e.g.
    /// `"stinger_notice"` → [`SoundId::StingerNotice`]. Returns `None` for an
    /// unknown name. This is the lookup a *data-driven* caller (a level's
    /// trigger→sound table in the host game) uses so authored content can
    /// name a sound as text without depending on the enum's declaration order.
    pub fn from_name(name: &str) -> Option<SoundId> {
        SoundId::ALL.into_iter().find(|id| id.name() == name)
    }
}

/// Number of synthesised sounds.
pub const SYNTH_SOUND_COUNT: usize = 56;

// A no-vibrato partial helper readability alias used inside the table below.
const NV: f32 = 0.0;

/// The recipes, in [`SoundId`] order. Numbers are copied verbatim from the
/// legacy `RECIPES[]` table so the rendered PCM matches the C build's intent.
static RECIPES: [Recipe; SYNTH_SOUND_COUNT] = [
    // 0 boot — rising two-tone chime, ~1.6 s, bright + clean.
    Recipe {
        name: "boot",
        duration_s: 1.6,
        master_gain: 0.55,
        env: Envelope {
            attack: 0.06,
            decay: 0.20,
            sustain: 0.75,
            release: 0.50,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 330.0, 660.0, 1.00, NV, NV), // rising root
            p(Wave::Sine, 495.0, 990.0, 0.45, NV, NV), // rising 5th
            p(Wave::Tri, 660.0, 1320.0, 0.20, NV, NV), // octave sparkle
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x1357,
    },
    // 1 switch_click — tiny percussive tick, ~0.06 s.
    Recipe {
        name: "switch_click",
        duration_s: 0.06,
        master_gain: 0.70,
        env: Envelope {
            attack: 0.001,
            decay: 0.010,
            sustain: 0.0,
            release: 0.045,
        },
        noise_mix: 0.35, // a little white noise gives the "click" its edge
        noise_lp: 0.0,
        partial_count: 1,
        partials: [
            p(Wave::Square, 1800.0, 1400.0, 1.00, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x2468,
    },
    // 2 interact — soft confirm blip, ~0.18 s, rounder than the switch.
    Recipe {
        name: "interact",
        duration_s: 0.18,
        master_gain: 0.55,
        env: Envelope {
            attack: 0.004,
            decay: 0.040,
            sustain: 0.30,
            release: 0.120,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 660.0, 880.0, 1.00, NV, NV), // small upward blip
            p(Wave::Tri, 1320.0, 1760.0, 0.25, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x9ABC,
    },
    // 3 door — low slow mechanical swell, ~2.6 s.
    Recipe {
        name: "door",
        duration_s: 2.6,
        master_gain: 0.60,
        env: Envelope {
            attack: 0.40,
            decay: 0.50,
            sustain: 0.80,
            release: 1.10,
        },
        noise_mix: 0.18, // low rumble of filtered noise under the tone
        noise_lp: 0.85,
        partial_count: 3,
        partials: [
            p(Wave::Saw, 70.0, 55.0, 1.00, NV, NV),     // heavy low body
            p(Wave::Square, 105.0, 82.0, 0.30, NV, NV), // grind harmonic
            p(Wave::Sine, 46.0, 40.0, 0.50, NV, NV),    // sub
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x55AA,
    },
    // 4 glitch — harsh digital burst, ~0.35 s. Mostly noise + a buzzy saw.
    Recipe {
        name: "glitch",
        duration_s: 0.35,
        master_gain: 0.80,
        env: Envelope {
            attack: 0.000,
            decay: 0.020,
            sustain: 0.55,
            release: 0.080,
        },
        noise_mix: 0.80, // dominated by raw white noise => harsh
        noise_lp: 0.0,
        partial_count: 2,
        partials: [
            p(Wave::Saw, 220.0, 90.0, 0.70, NV, NV),
            p(Wave::Square, 440.0, 160.0, 0.45, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        // legacy: 0xDEAD2u & 0xFFFFu
        seed: 0xDEAD2 & 0xFFFF,
    },
    // 5 voice — wordless tonal "thought voice", ~3.0 s, vowel-ish + vibrato.
    Recipe {
        name: "voice",
        duration_s: 3.0,
        master_gain: 0.50,
        env: Envelope {
            attack: 0.20,
            decay: 0.30,
            sustain: 0.85,
            release: 0.60,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 180.0, 165.0, 1.00, 5.0, 0.012), // fundamental
            p(Wave::Sine, 540.0, 495.0, 0.40, 5.0, 0.012), // 3rd harmonic
            p(Wave::Sine, 900.0, 825.0, 0.18, 5.0, 0.012), // 5th harmonic
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x7777,
    },
    // 6 pipe — soft flowing whoosh, ~2.0 s, filtered noise layer.
    Recipe {
        name: "pipe",
        duration_s: 2.0,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.40,
            decay: 0.30,
            sustain: 0.80,
            release: 0.60,
        },
        noise_mix: 0.95, // almost entirely dark filtered noise
        noise_lp: 0.92,
        partial_count: 1,
        partials: [
            p(Wave::Sine, 120.0, 100.0, 0.20, NV, NV), // faint tonal core
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0xF10F,
    },
    // 7 ambient — evolving low drone, ~10 s, designed to loop seamlessly.
    Recipe {
        name: "ambient",
        duration_s: 10.0,
        master_gain: 0.38,
        // near-flat envelope: tiny fade in/out so the loop seam is clean
        env: Envelope {
            attack: 0.40,
            decay: 0.00,
            sustain: 1.00,
            release: 0.40,
        },
        noise_mix: 0.10, // a whisper of dark room-tone noise
        noise_lp: 0.90,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 55.00, 55.00, 1.00, 0.07, 0.010), // deep drone A1
            p(Wave::Sine, 82.41, 82.41, 0.55, 0.11, 0.010), // fifth E2
            p(Wave::Tri, 110.0, 110.0, 0.28, 0.13, 0.008),  // octave A2
            p(Wave::Sine, 164.81, 164.81, 0.16, 0.05, 0.006), // shimmer E3
        ],
        seed: 0x0A1B,
    },
    // ── additive (indices 8..=13) ───────────────────────────────────────────
    // 8 footstep — short soft thud: low sine body + a quick filtered noise
    // burst, very fast attack, short release. ~0.12 s.
    Recipe {
        name: "footstep",
        duration_s: 0.12,
        master_gain: 0.55,
        env: Envelope {
            attack: 0.002,
            decay: 0.040,
            sustain: 0.20,
            release: 0.070,
        },
        noise_mix: 0.45, // dark filtered noise gives the soft "scuff"
        noise_lp: 0.80,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 150.0, 90.0, 1.00, NV, NV), // thud body
            p(Wave::Tri, 75.0, 55.0, 0.40, NV, NV),   // low sub
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0xF007,
    },
    // 9 ui_move — brief bright blip for menu navigation, ~0.05 s.
    Recipe {
        name: "ui_move",
        duration_s: 0.05,
        master_gain: 0.50,
        env: Envelope {
            attack: 0.001,
            decay: 0.012,
            sustain: 0.0,
            release: 0.035,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 880.0, 880.0, 1.00, NV, NV),  // clean A5 blip
            p(Wave::Tri, 1760.0, 1760.0, 0.18, NV, NV), // tiny sparkle
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x0B11,
    },
    // 10 ui_select — confirm tone, a touch longer & rounder, rises a step,
    // ~0.16 s. Reads as a positive "accepted" cue.
    Recipe {
        name: "ui_select",
        duration_s: 0.16,
        master_gain: 0.55,
        env: Envelope {
            attack: 0.003,
            decay: 0.030,
            sustain: 0.45,
            release: 0.110,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 660.0, 988.0, 1.00, NV, NV), // E5 -> B5 step
            p(Wave::Sine, 1320.0, 1976.0, 0.30, NV, NV), // octave gloss
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5E1E,
    },
    // 11 win_sting — short rising arpeggio/chord for solving a level, ~0.9 s.
    Recipe {
        name: "win_sting",
        duration_s: 0.90,
        master_gain: 0.52,
        env: Envelope {
            attack: 0.010,
            decay: 0.120,
            sustain: 0.65,
            release: 0.45,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 523.25, 523.25, 1.00, NV, NV),   // C5 root
            p(Wave::Sine, 659.25, 659.25, 0.70, NV, NV),   // E5 third
            p(Wave::Tri, 783.99, 783.99, 0.50, NV, NV),    // G5 fifth
            p(Wave::Sine, 1046.50, 1046.50, 0.35, NV, NV), // C6 octave
        ],
        seed: 0x1235,
    },
    // 12 ambient_cold — slow dark drone for the constellation theme, ~12 s.
    Recipe {
        name: "ambient_cold",
        duration_s: 12.0,
        master_gain: 0.34,
        env: Envelope {
            attack: 0.60,
            decay: 0.00,
            sustain: 1.00,
            release: 0.60,
        },
        noise_mix: 0.08, // whisper of very dark room-tone noise
        noise_lp: 0.93,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 49.00, 49.00, 1.00, 0.05, 0.008), // deep G1
            p(Wave::Sine, 58.27, 58.27, 0.55, 0.08, 0.008), // minor 3rd Bb1
            p(Wave::Tri, 98.00, 98.00, 0.22, 0.10, 0.006),  // octave G2
            p(Wave::Sine, 146.83, 146.83, 0.12, 0.04, 0.005), // cold shimmer D3
        ],
        seed: 0xC01D,
    },
    // 13 ambient_garden — warm gentle hum for the garden theme, ~12 s.
    Recipe {
        name: "ambient_garden",
        duration_s: 12.0,
        master_gain: 0.36,
        env: Envelope {
            attack: 0.60,
            decay: 0.00,
            sustain: 1.00,
            release: 0.60,
        },
        noise_mix: 0.06, // a little breathy mid-band noise (warmer than cold)
        noise_lp: 0.85,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 65.41, 65.41, 1.00, 0.06, 0.012), // warm C2
            p(Wave::Tri, 98.00, 98.00, 0.60, 0.09, 0.012),  // fifth G2
            p(Wave::Sine, 130.81, 130.81, 0.34, 0.12, 0.010), // octave C3
            p(Wave::Tri, 164.81, 164.81, 0.18, 0.05, 0.008), // major 3rd E3
        ],
        seed: 0x6A2D,
    },
    // ── rhythmic-audio additions (indices 14..=27; appended, never reorder) ──
    // 14 glitch_tick — micro bit-crush click, ~0.03 s. A tiny high square
    // transient with a touch of white noise: the "data tick" between beats.
    Recipe {
        name: "glitch_tick",
        duration_s: 0.03,
        master_gain: 0.55,
        env: Envelope {
            attack: 0.000,
            decay: 0.006,
            sustain: 0.0,
            release: 0.020,
        },
        noise_mix: 0.40,
        noise_lp: 0.0,
        partial_count: 1,
        partials: [
            p(Wave::Square, 2600.0, 2200.0, 1.00, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x6173,
    },
    // 15 data_burst — short filtered noise burst, ~0.16 s, like a packet of
    // corrupted data. Mostly band-limited noise with a faint saw edge.
    Recipe {
        name: "data_burst",
        duration_s: 0.16,
        master_gain: 0.50,
        env: Envelope {
            attack: 0.001,
            decay: 0.030,
            sustain: 0.35,
            release: 0.090,
        },
        noise_mix: 0.85,
        noise_lp: 0.55,
        partial_count: 2,
        partials: [
            p(Wave::Saw, 900.0, 600.0, 0.60, NV, NV),
            p(Wave::Square, 1500.0, 1100.0, 0.30, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x0DA7,
    },
    // 16 scanline — fast descending freq sweep, ~0.22 s: a CRT scanline whine
    // dropping through the spectrum.
    Recipe {
        name: "scanline",
        duration_s: 0.22,
        master_gain: 0.42,
        env: Envelope {
            attack: 0.002,
            decay: 0.040,
            sustain: 0.50,
            release: 0.120,
        },
        noise_mix: 0.12,
        noise_lp: 0.30,
        partial_count: 2,
        partials: [
            p(Wave::Saw, 3200.0, 400.0, 1.00, NV, NV),  // the sweep
            p(Wave::Sine, 1600.0, 200.0, 0.30, NV, NV), // sub-octave shadow
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5CA7,
    },
    // 17 stutter — short retriggered buzz, ~0.18 s: a square gated by a fast
    // tremolo-ish vibrato to read as a "stutter".
    Recipe {
        name: "stutter",
        duration_s: 0.18,
        master_gain: 0.48,
        env: Envelope {
            attack: 0.001,
            decay: 0.020,
            sustain: 0.60,
            release: 0.080,
        },
        noise_mix: 0.20,
        noise_lp: 0.10,
        partial_count: 2,
        partials: [
            p(Wave::Square, 520.0, 520.0, 1.00, 60.0, 0.40), // heavy fast wobble
            p(Wave::Saw, 1040.0, 1040.0, 0.30, 60.0, 0.40),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5717,
    },
    // 18 drone_layer_a — slow evolving mid drone, ~14 s, breathes via vibrato.
    // Layered under the ambient bed to thicken it.
    Recipe {
        name: "drone_layer_a",
        duration_s: 14.0,
        master_gain: 0.26,
        env: Envelope {
            attack: 1.20,
            decay: 0.00,
            sustain: 1.00,
            release: 1.20,
        },
        noise_mix: 0.05,
        noise_lp: 0.92,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 73.42, 73.42, 1.00, 0.06, 0.012),   // D2
            p(Wave::Tri, 110.00, 110.00, 0.45, 0.09, 0.010),  // A2
            p(Wave::Sine, 220.00, 220.00, 0.18, 0.04, 0.008), // A3 shimmer
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0xD4A1,
    },
    // 19 drone_layer_b — higher airy drone, ~14 s, even slower LFO.
    Recipe {
        name: "drone_layer_b",
        duration_s: 14.0,
        master_gain: 0.20,
        env: Envelope {
            attack: 1.50,
            decay: 0.00,
            sustain: 1.00,
            release: 1.50,
        },
        noise_mix: 0.04,
        noise_lp: 0.80,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 146.83, 146.83, 1.00, 0.03, 0.010), // D3
            p(Wave::Sine, 196.00, 196.00, 0.55, 0.05, 0.009), // G3
            p(Wave::Tri, 293.66, 293.66, 0.22, 0.07, 0.007),  // D4 air
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0xD4B2,
    },
    // ── tonal palette: A minor pentatonic, short plucked/bell-ish notes ──
    // All share an identical short pluck ADSR; only the pitch differs. ~0.7 s.
    tone("tone_a3", 220.00, 0x7041),
    tone("tone_c4", 261.63, 0x7042),
    tone("tone_d4", 293.66, 0x7043),
    tone("tone_e4", 329.63, 0x7044),
    tone("tone_g4", 392.00, 0x7045),
    tone("tone_a4", 440.00, 0x7046),
    tone("tone_c5", 523.25, 0x7047),
    tone("tone_d5", 587.33, 0x7048),
    // ── atmosphere drone moods (indices 28..=31) ─────────────────────────────
    // Long evolving beds, designed to be looped via kira's `loop_region` and
    // crossfaded against one another by `playback`. A long fade in/out keeps the
    // loop seam clean; near-flat sustain holds the bed. Detuned partials beat
    // slowly against each other; per-partial vibrato gives a slow LFO "breath";
    // a low-passed noise floor reads as room/machine air.
    //
    // 28 drone_cold_void — deepest + darkest: empty buffer, the void between
    // dead processes. Sub-bass root with a near-unison detune (~0.6 Hz beat) and
    // a hollow fifth; heavy dark noise floor. ~16 s.
    Recipe {
        name: "drone_cold_void",
        duration_s: 16.0,
        master_gain: 0.34,
        env: Envelope {
            attack: 2.00,
            decay: 0.00,
            sustain: 1.00,
            release: 2.00,
        },
        noise_mix: 0.12,
        noise_lp: 0.95, // very dark air
        partial_count: 4,
        partials: [
            p(Wave::Sine, 41.20, 41.20, 1.00, 0.04, 0.006), // E1 root
            p(Wave::Sine, 41.80, 41.80, 0.85, 0.05, 0.006), // detune → slow beat
            p(Wave::Sine, 61.74, 61.74, 0.40, 0.06, 0.005), // hollow 5th B1
            p(Wave::Tri, 82.41, 82.41, 0.14, 0.03, 0.004),  // faint octave E2
        ],
        seed: 0xC01D_4011 & 0xFFFF,
    },
    // 29 drone_server_hum — the dying machine: a mains-style hum (≈50 Hz) plus
    // its harmonics, slightly unstable. Mid-bright filtered noise = fan/disk air.
    // ~15 s.
    Recipe {
        name: "drone_server_hum",
        duration_s: 15.0,
        master_gain: 0.30,
        env: Envelope {
            attack: 1.50,
            decay: 0.00,
            sustain: 1.00,
            release: 1.50,
        },
        noise_mix: 0.16,
        noise_lp: 0.82,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 50.00, 50.00, 1.00, 0.12, 0.004), // mains hum
            p(Wave::Square, 100.0, 100.0, 0.22, 0.17, 0.006), // 2nd harmonic buzz
            p(Wave::Sine, 150.0, 150.0, 0.16, 0.23, 0.006), // 3rd harmonic
            p(Wave::Saw, 50.30, 50.30, 0.12, 0.09, 0.005),  // detuned grit
        ],
        seed: 0x5E20_4002 & 0xFFFF,
    },
    // 30 drone_organic_decay — warmer, breathing, rot/garden: a low minor triad
    // with a wider, slower LFO so it swells like something alive but failing.
    // Breathier mid noise. ~16 s.
    Recipe {
        name: "drone_organic_decay",
        duration_s: 16.0,
        master_gain: 0.30,
        env: Envelope {
            attack: 2.20,
            decay: 0.00,
            sustain: 1.00,
            release: 2.20,
        },
        noise_mix: 0.10,
        noise_lp: 0.70, // breathier than the cold beds
        partial_count: 4,
        partials: [
            p(Wave::Sine, 55.00, 55.00, 1.00, 0.07, 0.014), // A1 root, deep breath
            p(Wave::Tri, 65.41, 65.41, 0.55, 0.09, 0.012),  // minor 3rd C2
            p(Wave::Sine, 82.41, 82.41, 0.42, 0.11, 0.012), // 5th E2
            p(Wave::Sine, 110.0, 110.0, 0.20, 0.05, 0.010), // octave A2 shimmer
        ],
        seed: 0x0A6A_4003 & 0xFFFF,
    },
    // 31 drone_alarm_tension — rising dread: a beating minor-second cluster +
    // a higher unstable partial, no resolution. Faster, shallower LFO = nerves.
    // Brighter noise edge. ~14 s.
    Recipe {
        name: "drone_alarm_tension",
        duration_s: 14.0,
        master_gain: 0.28,
        env: Envelope {
            attack: 1.20,
            decay: 0.00,
            sustain: 1.00,
            release: 1.20,
        },
        noise_mix: 0.08,
        noise_lp: 0.55,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 110.0, 110.0, 1.00, 0.30, 0.006),   // A2
            p(Wave::Sine, 116.54, 116.54, 0.80, 0.34, 0.006), // Bb2 → harsh beat
            p(Wave::Tri, 220.0, 220.0, 0.30, 0.41, 0.005),    // A3 edge
            p(Wave::Saw, 233.08, 233.08, 0.16, 0.47, 0.005),  // Bb3 dissonant top
        ],
        seed: 0xA1A2_4004 & 0xFFFF,
    },
    // ── narrative trigger stingers (indices 32..=34) ─────────────────────────
    // Punctuation for story `[trigger]` fires. All are < 1.5 s by contract (see
    // `stinger_durations_within_bounds`). The aesthetic brief is the cold
    // MNEMOS/LETHE ambient pull: a facility noticing you, a memory pushing up
    // through static, data rotting — never a fanfare, never a resolved triad.
    //
    // 32 stinger_notice — "system notice": a detuned high sine pair, near-unison
    // so it beats (≈3.2 Hz) instead of ringing pure, plus a quiet triangle an
    // octave down for body. Instant attack, fast decay, thin tail: a cold ping
    // with no musical payload. A hair of low-passed noise gives it terminal
    // grit. ~0.45 s.
    Recipe {
        name: "stinger_notice",
        duration_s: 0.45,
        master_gain: 0.40,
        // Instant attack, fast decay, then a low tail long enough for one full
        // ≈3.2 Hz beat cycle of the detuned pair (≈310 ms) to be heard.
        env: Envelope {
            attack: 0.001,
            decay: 0.090,
            sustain: 0.14,
            release: 0.30,
        },
        noise_mix: 0.04,
        noise_lp: 0.65,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 987.77, 987.77, 1.00, NV, NV), // B5
            p(Wave::Sine, 991.00, 991.00, 0.55, NV, NV), // slight detune → beat
            p(Wave::Tri, 493.88, 493.88, 0.18, NV, NV),  // B4 body
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x57A1,
    },
    // 33 stinger_memory — "memory surfacing": a dark low-passed noise bed with a
    // long attack and no transient, out of which two detuned partials glide from
    // a smeared 11 Hz beat into near-unison (0.6 Hz) — the gesture of something
    // indistinct coming into focus. A bare fifth above (no third) keeps it cold.
    // Slow per-partial LFO = the pull. ~1.4 s, beat-quantized to a downbeat.
    Recipe {
        name: "stinger_memory",
        duration_s: 1.40,
        master_gain: 0.34,
        env: Envelope {
            attack: 0.55,
            decay: 0.30,
            sustain: 0.62,
            release: 0.42,
        },
        noise_mix: 0.40,
        noise_lp: 0.90, // dark static (≈740 Hz one-pole) the tone surfaces through
        partial_count: 3,
        partials: [
            p(Wave::Sine, 203.00, 220.00, 1.00, 0.9, 0.010), // → A3, focusing
            p(Wave::Sine, 214.00, 220.60, 0.85, 0.7, 0.008), // beat → near-unison
            p(Wave::Tri, 330.00, 329.63, 0.28, 0.5, 0.006),  // bare 5th E4
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x57A2,
    },
    // 34 stinger_corrupt — "corruption tick": a glitch-family transient, but
    // filtered and gestural rather than a bit-crush click. A square + saw sweep
    // collapsing downward (1400 → 220 Hz) under a fast 45 Hz wobble reads as a
    // retrigger/gate; band-limited noise (lp 0.45) is the corruption itself.
    // Routes to the accent track (distortion + delay). ~0.28 s.
    Recipe {
        name: "stinger_corrupt",
        duration_s: 0.28,
        master_gain: 0.46,
        env: Envelope {
            attack: 0.000,
            decay: 0.045,
            sustain: 0.30,
            release: 0.14,
        },
        noise_mix: 0.45,
        noise_lp: 0.62, // filtered (≈2 kHz), not raw white — the sweep stays audible
        partial_count: 2,
        partials: [
            p(Wave::Square, 1400.0, 220.0, 1.00, 45.0, 0.35), // collapsing sweep
            p(Wave::Saw, 700.0, 110.0, 0.42, 45.0, 0.35),     // sub-octave grit
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x57A3,
    },
    // 35 gaze_pulse — P2 "the gaze": a soft low-mid pulse that *beats*. Two
    // sines 2.5 Hz apart (146.83 = D3 against 149.3) drift in and out of phase
    // so the pulse never sits perfectly still — recognition, not threat. Slow
    // attack (no transient) + a whisper of dark noise for room. ~0.55 s.
    Recipe {
        name: "gaze_pulse",
        duration_s: 0.55,
        master_gain: 0.34,
        env: Envelope {
            attack: 0.070,
            decay: 0.110,
            sustain: 0.55,
            release: 0.30,
        },
        noise_mix: 0.06,
        noise_lp: 0.93,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 146.83, 146.83, 1.00, 0.9, 0.004), // D3 body
            p(Wave::Sine, 149.30, 149.30, 0.85, 0.9, 0.004), // detuned twin → beating
            p(Wave::Tri, 293.66, 293.66, 0.12, NV, NV),      // faint octave sheen
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x57A4,
    },
    // 36 river_wash — P3 "grief / LETHE": the river. Almost entirely
    // heavily-filtered noise (lp 0.965 ≈ a few hundred Hz) over a faint E2 that
    // sags a semitone-ish across the sound; long attack, long release, no
    // transient and no pitch centre to hold. ~3.2 s.
    Recipe {
        name: "river_wash",
        duration_s: 3.2,
        master_gain: 0.34,
        env: Envelope {
            attack: 0.90,
            decay: 0.55,
            sustain: 0.70,
            release: 1.20,
        },
        noise_mix: 0.92,
        noise_lp: 0.965,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 82.41, 77.78, 0.30, 0.13, 0.008), // sagging low tone
            p(Wave::Sine, 123.47, 116.54, 0.12, 0.09, 0.006), // hollow fifth
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x57A5,
    },
    // 37 heart_lub — P4 "closing": the downbeat thump. Low sine body dropping
    // 62 → 48 Hz with a sub under it and a trace of filtered noise for the
    // muscle; fast attack, short tail. Soft on purpose — near-silence is the
    // point, the heartbeat is what remains. ~0.30 s.
    Recipe {
        name: "heart_lub",
        duration_s: 0.30,
        master_gain: 0.50,
        env: Envelope {
            attack: 0.006,
            decay: 0.060,
            sustain: 0.25,
            release: 0.20,
        },
        noise_mix: 0.10,
        noise_lp: 0.93,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 62.0, 48.0, 1.00, NV, NV), // body
            p(Wave::Sine, 40.0, 33.0, 0.50, NV, NV), // sub
            p(Wave::Tri, 124.0, 96.0, 0.12, NV, NV), // faint knock
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x57A6,
    },
    // 38 heart_dub — P4 "closing": the answering thump, lower and quieter than
    // heart_lub so the pair reads as one beat of a heart rather than two hits.
    // ~0.26 s.
    Recipe {
        name: "heart_dub",
        duration_s: 0.26,
        master_gain: 0.38,
        env: Envelope {
            attack: 0.008,
            decay: 0.050,
            sustain: 0.20,
            release: 0.17,
        },
        noise_mix: 0.08,
        noise_lp: 0.94,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 55.0, 44.0, 1.00, NV, NV), // body
            p(Wave::Sine, 36.0, 30.0, 0.45, NV, NV), // sub
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x57A7,
    },
    // ── the puzzle SOUND GRAMMAR (indices 39..=55) ───────────────────────────
    // Four words, used consistently: RISING = progress, FALLING = regress /
    // refusal, CONSONANT (1, 3/2, 2, 5/4 partial ratios) = correct, DISSONANT
    // (minor-second pairs that beat) = blocked. `grammar_*` tests pin all four.
    //
    // 39 switch_on — a click whose tone rises a perfect fifth, A4 → E5, with a
    // consonant fifth riding above. The pipe opened; current flows. ~0.22 s.
    Recipe {
        name: "switch_on",
        duration_s: 0.22,
        master_gain: 0.50,
        env: Envelope {
            attack: 0.002,
            decay: 0.050,
            sustain: 0.45,
            release: 0.14,
        },
        noise_mix: 0.06, // a hair of click edge on the transient
        noise_lp: 0.50,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 440.0, 660.0, 1.00, NV, NV), // A4 → E5
            p(Wave::Sine, 660.0, 990.0, 0.35, NV, NV), // the fifth above, rising with it
            p(Wave::Tri, 880.0, 1320.0, 0.12, NV, NV), // octave sheen
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A01,
    },
    // 40 switch_off — the exact inverse gesture: E5 → A4, a lower octave shadow,
    // quieter and darker. Closing a pipe is not failure, but it is not progress.
    Recipe {
        name: "switch_off",
        duration_s: 0.24,
        master_gain: 0.42,
        env: Envelope {
            attack: 0.002,
            decay: 0.060,
            sustain: 0.40,
            release: 0.15,
        },
        noise_mix: 0.06,
        noise_lp: 0.70,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 660.0, 440.0, 1.00, NV, NV), // E5 → A4
            p(Wave::Sine, 330.0, 220.0, 0.40, NV, NV), // octave below, falling
            p(Wave::Tri, 990.0, 660.0, 0.08, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A02,
    },
    // 41 interact_empty — a dull thud with no pitch direction: the press hit air.
    Recipe {
        name: "interact_empty",
        duration_s: 0.09,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.001,
            decay: 0.025,
            sustain: 0.25,
            release: 0.055,
        },
        noise_mix: 0.50,
        noise_lp: 0.88,
        partial_count: 1,
        partials: [
            p(Wave::Sine, 160.0, 158.0, 1.00, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A03,
    },
    // 42 refuse_power — brownout: two saws a minor second apart (Bb3 / B3 →
    // F3 / F#3) sag together; the pair beats at ~14 Hz. Dissonant + falling =
    // "blocked": the grid has no power left for this pipe.
    Recipe {
        name: "refuse_power",
        duration_s: 0.42,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.003,
            decay: 0.080,
            sustain: 0.55,
            release: 0.20,
        },
        noise_mix: 0.08,
        noise_lp: 0.70,
        partial_count: 3,
        partials: [
            p(Wave::Saw, 233.08, 174.61, 1.00, NV, NV),   // Bb3 → F3
            p(Wave::Saw, 246.94, 185.00, 0.85, NV, NV),   // B3 → F#3: minor 2nd above
            p(Wave::Square, 116.54, 87.31, 0.30, NV, NV), // sub buzz
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A04,
    },
    // 43 refuse_cycle — the loop would close: a stuttering (30 Hz wobble)
    // semitone clash (F#5 / G5) collapsing an octave. Accent track.
    Recipe {
        name: "refuse_cycle",
        duration_s: 0.34,
        master_gain: 0.38,
        env: Envelope {
            attack: 0.001,
            decay: 0.050,
            sustain: 0.45,
            release: 0.15,
        },
        noise_mix: 0.22,
        noise_lp: 0.50,
        partial_count: 2,
        partials: [
            p(Wave::Square, 739.99, 369.99, 1.00, 30.0, 0.25), // F#5 → F#4
            p(Wave::Saw, 783.99, 392.00, 0.60, 30.0, 0.25),    // G5 → G4: semitone clash
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A05,
    },
    // 44 node_lit — a glassy chime that scoops UP a whole tone into E4, with
    // consonant octave + twelfth partials. The cue layer re-pitches it up a major
    // pentatonic ladder as more of the network lights. ~0.9 s ring.
    Recipe {
        name: "node_lit",
        duration_s: 0.90,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.003,
            decay: 0.25,
            sustain: 0.25,
            release: 0.55,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 293.66, 329.63, 1.00, NV, NV), // D4 → E4 scoop up
            p(Wave::Sine, 587.33, 659.26, 0.30, NV, NV), // octave
            p(Wave::Sine, 880.99, 988.88, 0.12, NV, NV), // twelfth
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A06,
    },
    // 45 node_unlit — the chime falls a minor third (E4 → C#4) under a
    // minor-second shadow (F4 → D4): a light going out, slightly wrong.
    Recipe {
        name: "node_unlit",
        duration_s: 0.70,
        master_gain: 0.36,
        env: Envelope {
            attack: 0.004,
            decay: 0.20,
            sustain: 0.25,
            release: 0.40,
        },
        noise_mix: 0.03,
        noise_lp: 0.90,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 329.63, 277.18, 1.00, NV, NV), // E4 → C#4
            p(Wave::Sine, 349.23, 293.66, 0.35, NV, NV), // F4 → D4: minor-2nd shadow
            p(Wave::Tri, 659.26, 554.37, 0.10, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A07,
    },
    // 46 circuit_resolve — the network is satisfied: an A-major bloom (A3, E4,
    // A4, C#5) with a slow attack. The one fully resolved chord the puzzle owns.
    Recipe {
        name: "circuit_resolve",
        duration_s: 1.60,
        master_gain: 0.36,
        env: Envelope {
            attack: 0.12,
            decay: 0.40,
            sustain: 0.60,
            release: 0.80,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 220.00, 220.00, 1.00, 0.4, 0.003),
            p(Wave::Sine, 329.63, 329.63, 0.70, 0.5, 0.003),
            p(Wave::Tri, 440.00, 440.00, 0.45, 0.3, 0.003),
            p(Wave::Sine, 554.37, 554.37, 0.35, 0.6, 0.003),
        ],
        seed: 0x5A08,
    },
    // 47 aim_tick — the crosshair found a switch: a tiny rising blip, very quiet.
    Recipe {
        name: "aim_tick",
        duration_s: 0.04,
        master_gain: 0.24,
        env: Envelope {
            attack: 0.001,
            decay: 0.010,
            sustain: 0.30,
            release: 0.022,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 1760.0, 1975.5, 1.00, NV, NV), // A6 → B6
            p(Wave::Tri, 3520.0, 3951.0, 0.12, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A09,
    },
    // 48 note_read — a soft page-open swish with a gentle rising fourth under it.
    Recipe {
        name: "note_read",
        duration_s: 0.50,
        master_gain: 0.32,
        env: Envelope {
            attack: 0.020,
            decay: 0.10,
            sustain: 0.35,
            release: 0.30,
        },
        noise_mix: 0.35,
        noise_lp: 0.60,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 587.33, 783.99, 1.00, NV, NV), // D5 → G5
            p(Wave::Tri, 1174.66, 1567.98, 0.18, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A0A,
    },
    // 49 portal_warp — space bending: three octave-spaced partials sweeping up
    // three octaves together over a dark noise rush. ~1.1 s.
    Recipe {
        name: "portal_warp",
        duration_s: 1.10,
        master_gain: 0.34,
        env: Envelope {
            attack: 0.15,
            decay: 0.30,
            sustain: 0.60,
            release: 0.50,
        },
        noise_mix: 0.30,
        noise_lp: 0.80,
        partial_count: 3,
        partials: [
            p(Wave::Sine, 220.0, 1760.0, 1.00, 6.0, 0.05),
            p(Wave::Saw, 110.0, 880.0, 0.35, 6.0, 0.05),
            p(Wave::Sine, 55.0, 440.0, 0.40, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A0B,
    },
    // 50 storey_up — a low swell rising a fifth (G2 → D3).
    Recipe {
        name: "storey_up",
        duration_s: 0.70,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.08,
            decay: 0.15,
            sustain: 0.60,
            release: 0.35,
        },
        noise_mix: 0.12,
        noise_lp: 0.90,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 98.0, 146.83, 1.00, NV, NV),
            p(Wave::Tri, 196.0, 293.66, 0.35, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A0C,
    },
    // 51 storey_down — the same swell falling a fifth (D3 → G2).
    Recipe {
        name: "storey_down",
        duration_s: 0.70,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.08,
            decay: 0.15,
            sustain: 0.60,
            release: 0.35,
        },
        noise_mix: 0.12,
        noise_lp: 0.90,
        partial_count: 2,
        partials: [
            p(Wave::Sine, 146.83, 98.0, 1.00, NV, NV),
            p(Wave::Tri, 293.66, 196.0, 0.35, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A0D,
    },
    // 52 footstep_metal — a grate clang: inharmonic ringing partials over a
    // bright noise tick. ~0.14 s.
    Recipe {
        name: "footstep_metal",
        duration_s: 0.14,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.001,
            decay: 0.030,
            sustain: 0.25,
            release: 0.10,
        },
        noise_mix: 0.35,
        noise_lp: 0.40,
        partial_count: 3,
        partials: [
            p(Wave::Square, 420.0, 395.0, 0.45, NV, NV),
            p(Wave::Sine, 1253.0, 1190.0, 0.60, NV, NV), // inharmonic ring
            p(Wave::Tri, 2117.0, 2050.0, 0.22, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A0E,
    },
    // 53 footstep_wet — a gloop: dark noise over a body that drops fast.
    Recipe {
        name: "footstep_wet",
        duration_s: 0.16,
        master_gain: 0.42,
        env: Envelope {
            attack: 0.002,
            decay: 0.050,
            sustain: 0.30,
            release: 0.10,
        },
        noise_mix: 0.70,
        noise_lp: 0.78,
        partial_count: 1,
        partials: [
            p(Wave::Sine, 320.0, 140.0, 1.00, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A0F,
    },
    // 54 footstep_soft — a muffled crunch: almost all very dark noise.
    Recipe {
        name: "footstep_soft",
        duration_s: 0.13,
        master_gain: 0.40,
        env: Envelope {
            attack: 0.004,
            decay: 0.040,
            sustain: 0.30,
            release: 0.08,
        },
        noise_mix: 0.85,
        noise_lp: 0.93,
        partial_count: 1,
        partials: [
            p(Wave::Sine, 110.0, 80.0, 1.00, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed: 0x5A10,
    },
    // 55 first_light — a colour shard recovered: a slow A-major bloom that
    // breathes (sub-Hz vibrato) over a whisper of air. ~2.4 s, lands on a bar.
    Recipe {
        name: "first_light",
        duration_s: 2.40,
        master_gain: 0.34,
        env: Envelope {
            attack: 0.60,
            decay: 0.40,
            sustain: 0.70,
            release: 1.20,
        },
        noise_mix: 0.05,
        noise_lp: 0.95,
        partial_count: 4,
        partials: [
            p(Wave::Sine, 220.00, 220.00, 1.00, 0.3, 0.004),
            p(Wave::Sine, 329.63, 329.63, 0.70, 0.4, 0.004),
            p(Wave::Sine, 440.00, 440.00, 0.50, 0.2, 0.004),
            p(Wave::Tri, 554.37, 554.37, 0.35, 0.5, 0.004),
        ],
        seed: 0x5A11,
    },
];

/// Build a short plucked/bell-ish tonal-palette recipe at `fundamental` Hz.
/// `const fn` so it can populate the static `RECIPES` table. A sine fundamental
/// plus a quieter octave + twelfth gives a bell-like ping; the fast-attack /
/// medium-release ADSR reads as a pluck.
const fn tone(name: &'static str, fundamental: f32, seed: u32) -> Recipe {
    Recipe {
        name,
        duration_s: 0.7,
        master_gain: 0.45,
        env: Envelope {
            attack: 0.004,
            decay: 0.18,
            sustain: 0.30,
            release: 0.40,
        },
        noise_mix: 0.0,
        noise_lp: 0.0,
        partial_count: 3,
        partials: [
            p(Wave::Sine, fundamental, fundamental, 1.00, NV, NV),
            p(
                Wave::Sine,
                fundamental * 2.0,
                fundamental * 2.0,
                0.35,
                NV,
                NV,
            ),
            p(
                Wave::Tri,
                fundamental * 3.0,
                fundamental * 3.0,
                0.14,
                NV,
                NV,
            ),
            p(Wave::Sine, 0.0, 0.0, 0.0, NV, NV),
        ],
        seed,
    }
}

// ───────────────────────── render one recipe ────────────────────────────────

/// Clamp a float sample to `[-1, 1]`. The rodio [`crate::playback`] layer
/// consumes `f32` directly, so unlike the legacy build there is no int16
/// quantisation step in the hot path — but [`SynthSound::to_i16`] is provided
/// as a stable int16 view for determinism hashing (build-vs-build; no committed
/// golden baseline).
#[inline]
fn clamp_unit(v: f32) -> f32 {
    v.clamp(-1.0, 1.0)
}

/// Quantise a `[-1, 1]` float to int16 exactly as legacy `to_int16` did
/// (round-half-away-from-zero, clamp). Used by tests + the i16 accessor.
#[inline]
fn to_i16(v: f32) -> i16 {
    let v = clamp_unit(v);
    let scaled = v * 32767.0;
    let s = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    };
    s.clamp(-32768, 32767) as i16
}

/// One rendered sound: mono f32 PCM at [`SAMPLE_RATE`].
#[derive(Clone, Debug)]
pub struct SynthSound {
    /// Interleaved mono samples in `[-1, 1]`.
    pub frames: Vec<f32>,
    /// Always [`SAMPLE_RATE`].
    pub sample_rate: u32,
}

impl SynthSound {
    /// Number of frames (== sample count, mono).
    #[inline]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// The same PCM quantised to int16 — a stable int16 view used for
    /// determinism hashing (build-vs-build; not pinned to a golden baseline).
    pub fn to_i16(&self) -> Vec<i16> {
        self.frames.iter().map(|&v| to_i16(v)).collect()
    }
}

/// Render one recipe into a fresh f32 buffer. A faithful port of legacy
/// `render_recipe` (phase accumulation, glide, vibrato, noise LP, ADSR).
fn render(rc: &Recipe) -> SynthSound {
    let sr = SAMPLE_RATE;
    let mut n = (rc.duration_s as f64 * sr as f64 + 0.5) as usize;
    if n == 0 {
        n = 1;
    }

    let mut buf = Vec::with_capacity(n);

    let mut lcg = Lcg(rc.seed);
    let mut noise_lp_state: f32 = 0.0;

    // Per-partial running phase (in cycles), as f64 to match the C build.
    let mut phase = [0.0f64; MAX_PARTIALS];

    for i in 0..n {
        let frac = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };

        let mut sample = 0.0f32;
        let mut gain_sum = 0.0f32;

        for (pi, pt) in rc.partials.iter().enumerate().take(rc.partial_count) {
            // Linear frequency glide across the sound.
            let mut f = pt.freq0 as f64 + (pt.freq1 - pt.freq0) as f64 * frac;

            // Optional vibrato (sine LFO on frequency).
            if pt.vibrato_hz > 0.0 && pt.vibrato_depth > 0.0 {
                let tsec = i as f64 / sr as f64;
                let lfo = (TWO_PI as f64 * pt.vibrato_hz as f64 * tsec).sin();
                f *= 1.0 + pt.vibrato_depth as f64 * lfo;
            }

            phase[pi] += f / sr as f64;
            phase[pi] -= phase[pi].floor(); // keep bounded

            sample += pt.gain * osc(pt.wave, phase[pi] as f32);
            gain_sum += pt.gain;
        }
        if gain_sum > 0.0 {
            sample /= gain_sum; // normalise partial mix
        }

        // Noise layer (optionally low-passed for a darker, pinker tone).
        if rc.noise_mix > 0.0 {
            let wn = noise(&mut lcg);
            let nv = if rc.noise_lp > 0.0 {
                // one-pole LP: y += (1 - a) * (x - y); store state.
                noise_lp_state += (1.0 - rc.noise_lp) * (wn - noise_lp_state);
                // low-passing attenuates amplitude; compensate a touch.
                noise_lp_state * 2.0
            } else {
                wn
            };
            sample = (1.0 - rc.noise_mix) * sample + rc.noise_mix * nv;
        }

        let envg = env(&rc.env, i, n, sr);
        let v = sample * envg * rc.master_gain;
        buf.push(clamp_unit(v));
    }

    SynthSound {
        frames: buf,
        sample_rate: sr,
    }
}

// ─────────────────────────────── bank ───────────────────────────────────────

/// All synthesised sounds, indexed by [`SoundId`]. Built once at startup.
#[derive(Clone, Debug)]
pub struct SynthBank {
    sounds: Vec<SynthSound>,
}

impl SynthBank {
    /// Render every recipe. Deterministic: same build → identical buffers
    /// every call (determinism-tested build-vs-build; no golden baseline).
    ///
    /// Parallel fan-out. Recipe rendering is embarrassingly parallel: each
    /// [`render`] call is a pure function of its own `&Recipe` (its own private
    /// `Lcg` and phase accumulators, no shared state across sounds), and
    /// several recipes render ten-plus seconds of audio (see the ambient
    /// drones below). Building the whole bank serially on one thread was
    /// measured at roughly 570ms in a release build, the dominant cost of a
    /// host game's own app startup (roughly 0.6s of Bevy Startup runs before
    /// the alt screen appears). A
    /// `std::thread::scope` work-stealing fan-out over an atomic index, the
    /// same pattern `re_imgqa::run::run_level` uses, cuts wall time down to
    /// roughly the single longest recipe's render time instead of the sum of
    /// all of them. Deterministic regardless of thread count or scheduling:
    /// every result is written into a fixed slot keyed by recipe index, never
    /// by completion order, so `deterministic_same_recipe_same_bytes`
    /// (build-vs-build, unchanged) still holds.
    pub fn build() -> SynthBank {
        let n = RECIPES.len();
        let slots: Vec<Mutex<Option<SynthSound>>> = (0..n).map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);
        // Cap threads at the work available — spawning more workers than
        // recipes just adds join overhead for threads that steal nothing.
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .min(n.max(1));
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let next = &next;
                let slots = &slots;
                scope.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let sound = render(&RECIPES[i]);
                    let mut slot = slots[i].lock().unwrap_or_else(|p| p.into_inner());
                    *slot = Some(sound);
                });
            }
        });
        let sounds = slots
            .into_iter()
            .map(|slot| {
                slot.into_inner()
                    .unwrap_or_else(|p| p.into_inner())
                    .expect("every slot is rendered exactly once by the work-stealing loop above")
            })
            .collect();
        SynthBank { sounds }
    }

    /// Borrow the rendered PCM for a sound.
    #[inline]
    pub fn get(&self, id: SoundId) -> &SynthSound {
        &self.sounds[id.index()]
    }

    /// Number of sounds in the bank (== [`SYNTH_SOUND_COUNT`]).
    #[inline]
    pub fn len(&self) -> usize {
        self.sounds.len()
    }

    /// Always non-empty in practice; provided to satisfy clippy.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sounds.is_empty()
    }
}

impl Default for SynthBank {
    fn default() -> Self {
        SynthBank::build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: collect a deterministic 64-bit FNV-1a hash of the i16 bytes.
    fn fnv1a_i16(samples: &[i16]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &s in samples {
            for b in s.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Peak absolute sample + the frame index it occurs at.
    fn peak_of(frames: &[f32]) -> (f32, usize) {
        let mut best = (0.0f32, 0usize);
        for (i, &v) in frames.iter().enumerate() {
            if v.abs() > best.0 {
                best = (v.abs(), i);
            }
        }
        best
    }

    /// RMS energy of a buffer (0.0 for an empty one).
    fn rms_of(frames: &[f32]) -> f64 {
        if frames.is_empty() {
            return 0.0;
        }
        let sum: f64 = frames.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        (sum / frames.len() as f64).sqrt()
    }

    #[test]
    fn every_sound_builds_non_empty() {
        let bank = SynthBank::build();
        assert_eq!(bank.len(), SYNTH_SOUND_COUNT);
        for id in SoundId::ALL {
            let s = bank.get(id);
            assert!(
                s.frame_count() > 0,
                "{} rendered an empty buffer",
                id.name()
            );
            assert_eq!(s.sample_rate, SAMPLE_RATE);
        }
    }

    #[test]
    fn no_f32_or_i16_clipping() {
        // Every float sample must be inside [-1, 1]; every quantised i16 must
        // be a true int16 (never the overflow wrap). clamp_unit guarantees the
        // former; check it holds and that to_i16 stays in range.
        let bank = SynthBank::build();
        for id in SoundId::ALL {
            let s = bank.get(id);
            for &v in &s.frames {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "{}: float sample {v} out of [-1, 1]",
                    id.name()
                );
            }
            for q in s.to_i16() {
                // i16 by construction; assert the round-trip never produced the
                // sentinel that would indicate an overflow bug.
                assert!((-32768..=32767).contains(&(q as i32)));
            }
        }
    }

    #[test]
    fn deterministic_same_recipe_same_bytes() {
        // Two independent builds must be byte-identical (seeded LCG, no clock).
        let a = SynthBank::build();
        let b = SynthBank::build();
        for id in SoundId::ALL {
            let ha = fnv1a_i16(&a.get(id).to_i16());
            let hb = fnv1a_i16(&b.get(id).to_i16());
            assert_eq!(ha, hb, "{} not deterministic across builds", id.name());
            // And the raw f32 frames must match exactly too.
            assert_eq!(a.get(id).frames, b.get(id).frames);
        }
    }

    #[test]
    fn adsr_monotonic_in_attack_nonincreasing_in_release() {
        // A clear ADSR with all four segments present at 44.1 kHz.
        let e = Envelope {
            attack: 0.10,
            decay: 0.05,
            sustain: 0.5,
            release: 0.10,
        };
        let total = SAMPLE_RATE as usize; // 1.0 s
        let sr = SAMPLE_RATE;

        // Attack region: indices [0, attack*sr). Must be non-decreasing and
        // span 0 -> ~1.
        let attack_end = (e.attack * sr as f32) as usize;
        let mut prev = env(&e, 0, total, sr);
        assert!(prev <= 1e-3, "attack should start near 0, got {prev}");
        for i in 1..attack_end {
            let g = env(&e, i, total, sr);
            assert!(
                g + 1e-6 >= prev,
                "attack not monotonic at {i}: {g} < {prev}"
            );
            prev = g;
        }
        assert!(prev > 0.9, "attack should reach ~1, got {prev}");

        // Release region: indices [(total - release*sr), total). Non-increasing
        // down to ~0.
        let rel_start = total - (e.release * sr as f32) as usize;
        let mut prev = env(&e, rel_start, total, sr);
        for i in (rel_start + 1)..total {
            let g = env(&e, i, total, sr);
            assert!(
                g <= prev + 1e-6,
                "release not non-increasing at {i}: {g} > {prev}"
            );
            prev = g;
        }
        assert!(prev <= 1e-2, "release should end near 0, got {prev}");
    }

    #[test]
    fn osc_known_phase_values() {
        // Sine: 0 at phase 0, +1 at 1/4, 0 at 1/2.
        assert!(osc(Wave::Sine, 0.0).abs() < 1e-6);
        assert!((osc(Wave::Sine, 0.25) - 1.0).abs() < 1e-6);
        assert!(osc(Wave::Sine, 0.5).abs() < 1e-6);

        // Triangle: 0 at phase 0, +1 peak at 1/4, 0 at 1/2, -1 trough at 3/4.
        assert!(osc(Wave::Tri, 0.0).abs() < 1e-6);
        assert!((osc(Wave::Tri, 0.25) - 1.0).abs() < 1e-6);
        assert!(osc(Wave::Tri, 0.5).abs() < 1e-6);
        assert!((osc(Wave::Tri, 0.75) + 1.0).abs() < 1e-6);

        // Square: +1 in the first half, -1 in the second.
        assert_eq!(osc(Wave::Square, 0.1), 1.0);
        assert_eq!(osc(Wave::Square, 0.9), -1.0);

        // Saw: -1 at phase 0, ~+1 approaching 1.
        assert!((osc(Wave::Saw, 0.0) + 1.0).abs() < 1e-6);
        assert!((osc(Wave::Saw, 0.5)).abs() < 1e-6);

        // Noise via osc() is silent (non-periodic; use noise()).
        assert_eq!(osc(Wave::Noise, 0.3), 0.0);
    }

    #[test]
    fn new_recipes_present_and_named() {
        // The rhythmic-audio additions + the 4 atmospheric drones exist, render
        // non-empty, and keep the names the policy/director tables expect.
        let bank = SynthBank::build();
        assert_eq!(SYNTH_SOUND_COUNT, 56);
        let expected = [
            (SoundId::GlitchTick, "glitch_tick"),
            (SoundId::DataBurst, "data_burst"),
            (SoundId::Scanline, "scanline"),
            (SoundId::Stutter, "stutter"),
            (SoundId::DroneLayerA, "drone_layer_a"),
            (SoundId::DroneLayerB, "drone_layer_b"),
            (SoundId::ToneA3, "tone_a3"),
            (SoundId::ToneD5, "tone_d5"),
        ];
        for (id, name) in expected {
            assert_eq!(id.name(), name);
            assert!(bank.get(id).frame_count() > 0);
        }
    }

    #[test]
    fn new_recipes_are_deterministic_and_unclipped() {
        // Same determinism + no-clip guarantee as the originals, restricted to
        // the appended ids (indices 14..28).
        let a = SynthBank::build();
        let b = SynthBank::build();
        for id in &SoundId::ALL[14..] {
            let id = *id;
            assert_eq!(
                fnv1a_i16(&a.get(id).to_i16()),
                fnv1a_i16(&b.get(id).to_i16()),
                "{} not deterministic",
                id.name()
            );
            for &v in &a.get(id).frames {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "{}: sample {v} out of range",
                    id.name()
                );
            }
        }
    }

    #[test]
    fn tonal_palette_is_distinct_pitches() {
        // The palette notes must not be byte-identical to each other (different
        // fundamentals → different PCM), so the director's pitch choice matters.
        let bank = SynthBank::build();
        let mut hashes = std::collections::HashSet::new();
        for id in SoundId::TONAL_PALETTE {
            let h = fnv1a_i16(&bank.get(id).to_i16());
            assert!(hashes.insert(h), "{} duplicated another tone", id.name());
        }
    }

    #[test]
    fn drone_moods_are_present_nonsilent_distinct_deterministic() {
        // The four atmosphere drone beds must each render a long, audibly
        // non-silent buffer, be distinct from one another (different moods), and
        // be byte-identical across builds (seeded). This is the contract the
        // ambience/crossfade layer relies on.
        let a = SynthBank::build();
        let b = SynthBank::build();
        assert_eq!(SoundId::DRONE_MOODS.len(), 4);

        let mut hashes = std::collections::HashSet::new();
        for id in SoundId::DRONE_MOODS {
            let s = a.get(id);
            // Long bed: at least several seconds of audio.
            assert!(
                s.frame_count() > 10 * SAMPLE_RATE as usize,
                "{} is too short to be a drone bed",
                id.name()
            );
            // Non-silent: RMS energy is clearly above the noise floor.
            let rms = rms_of(&s.frames);
            assert!(
                rms > 0.01,
                "{} is effectively silent (rms {rms})",
                id.name()
            );
            // Deterministic across builds.
            assert_eq!(
                fnv1a_i16(&s.to_i16()),
                fnv1a_i16(&b.get(id).to_i16()),
                "{} not deterministic",
                id.name()
            );
            // Distinct mood (different PCM hash from the others).
            assert!(
                hashes.insert(fnv1a_i16(&s.to_i16())),
                "{} duplicated another drone mood",
                id.name()
            );
            // No clipping.
            for &v in &s.frames {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "{}: sample {v} clipped",
                    id.name()
                );
            }
        }
    }

    // ───────────────────────── narrative stingers ────────────────────────────

    #[test]
    fn stingers_present_and_named() {
        let bank = SynthBank::build();
        let expected = [
            (SoundId::StingerNotice, "stinger_notice"),
            (SoundId::StingerMemory, "stinger_memory"),
            (SoundId::StingerCorrupt, "stinger_corrupt"),
        ];
        assert_eq!(SoundId::STINGERS.len(), expected.len());
        for ((id, name), &canon) in expected.into_iter().zip(SoundId::STINGERS.iter()) {
            assert_eq!(id, canon, "STINGERS order drifted from the enum order");
            assert_eq!(id.name(), name);
            assert!(bank.get(id).frame_count() > 0, "{name} rendered empty");
        }
        // A stinger is punctuation, not ambient texture: the generative director
        // picks from GLITCH_FAMILY / TONAL_PALETTE, and no stinger may be there.
        for id in SoundId::STINGERS {
            assert!(
                !SoundId::GLITCH_FAMILY.contains(&id),
                "{} leaked into the director's glitch palette",
                id.name()
            );
            assert!(!SoundId::TONAL_PALETTE.contains(&id));
            assert!(!SoundId::DRONE_MOODS.contains(&id));
        }
    }

    #[test]
    fn stinger_durations_within_bounds() {
        // Design contract: every stinger is short punctuation — audible but well
        // under 1.5 s, so it never smears across the beat it lands on.
        let bank = SynthBank::build();
        let max_frames = (1.5 * SAMPLE_RATE as f32) as usize;
        let min_frames = (0.05 * SAMPLE_RATE as f32) as usize;
        for id in SoundId::STINGERS {
            let n = bank.get(id).frame_count();
            assert!(
                n < max_frames,
                "{} is {n} frames (>= 1.5 s) — too long for a stinger",
                id.name()
            );
            assert!(
                n > min_frames,
                "{} is {n} frames — too short to read",
                id.name()
            );
        }
    }

    #[test]
    fn stingers_deterministic_same_seed_same_pcm_hash() {
        // Same seeded recipe → byte-identical PCM, and the three stingers are
        // distinct from one another and from every pre-existing sound.
        let a = SynthBank::build();
        let b = SynthBank::build();
        let mut all: std::collections::HashSet<u64> = SoundId::ALL
            .into_iter()
            .filter(|id| !SoundId::STINGERS.contains(id))
            .map(|id| fnv1a_i16(&a.get(id).to_i16()))
            .collect();
        for id in SoundId::STINGERS {
            let ha = fnv1a_i16(&a.get(id).to_i16());
            assert_eq!(
                ha,
                fnv1a_i16(&b.get(id).to_i16()),
                "{} not deterministic across builds",
                id.name()
            );
            assert_eq!(a.get(id).frames, b.get(id).frames);
            assert!(all.insert(ha), "{} duplicated another sound", id.name());
        }
    }

    #[test]
    fn stingers_audible_with_headroom_and_never_clamped() {
        // Non-silent, and the peak stays strictly inside [-1, 1] — i.e. the
        // gain staging leaves headroom and `clamp_unit` never actually fired
        // (a clamped stinger would be audible distortion, not a design choice).
        let bank = SynthBank::build();
        for id in SoundId::STINGERS {
            let s = bank.get(id);
            let (peak, _) = peak_of(&s.frames);
            assert!(
                peak > 0.05,
                "{} peaks at only {peak} — inaudible",
                id.name()
            );
            assert!(
                peak < 0.99,
                "{} peaks at {peak} — clipping/clamped",
                id.name()
            );
            let rms = rms_of(&s.frames);
            assert!(
                rms > 0.005,
                "{} is effectively silent (rms {rms})",
                id.name()
            );
        }
    }

    #[test]
    fn stinger_shapes_match_their_design() {
        // The three stingers must be *shaped* differently, not three variants of
        // one click: notice/corrupt are transients (peak in the opening), memory
        // is a swell with no transient (peak well after the start, and its first
        // 20 ms is far quieter than its body).
        let bank = SynthBank::build();

        for id in [SoundId::StingerNotice, SoundId::StingerCorrupt] {
            let s = bank.get(id);
            let (_, at) = peak_of(&s.frames);
            let limit = s.frame_count() / 5; // first 20 %
            assert!(
                at < limit,
                "{} peaks at frame {at} of {} — not a transient",
                id.name(),
                s.frame_count()
            );
        }

        let s = bank.get(SoundId::StingerMemory);
        let (peak, at) = peak_of(&s.frames);
        let start = s.frame_count() * 15 / 100;
        assert!(
            at > start,
            "stinger_memory peaks at frame {at} of {} — that is a transient, not a swell",
            s.frame_count()
        );
        let head = (0.020 * SAMPLE_RATE as f32) as usize;
        let head_peak = peak_of(&s.frames[..head]).0;
        assert!(
            head_peak < 0.15 * peak,
            "stinger_memory opens at {head_peak} vs peak {peak} — attack is too abrupt"
        );
    }

    // ───────────────────────── movement motifs ──────────────────────────────

    #[test]
    fn movement_motifs_present_named_and_out_of_the_ambient_palettes() {
        let bank = SynthBank::build();
        let expected = [
            (SoundId::GazePulse, "gaze_pulse"),
            (SoundId::RiverWash, "river_wash"),
            (SoundId::HeartLub, "heart_lub"),
            (SoundId::HeartDub, "heart_dub"),
        ];
        assert_eq!(SoundId::MOVEMENT_MOTIFS.len(), expected.len());
        for ((id, name), &canon) in expected.into_iter().zip(SoundId::MOVEMENT_MOTIFS.iter()) {
            assert_eq!(
                id, canon,
                "MOVEMENT_MOTIFS order drifted from the enum order"
            );
            assert_eq!(id.name(), name);
            assert!(bank.get(id).frame_count() > 0, "{name} rendered empty");
        }
        // A motif belongs to a theme, not to the ambient texture palettes nor to
        // the story stingers.
        for id in SoundId::MOVEMENT_MOTIFS {
            assert!(
                !SoundId::GLITCH_FAMILY.contains(&id),
                "{} leaked into the glitch palette",
                id.name()
            );
            assert!(!SoundId::TONAL_PALETTE.contains(&id));
            assert!(!SoundId::DRONE_MOODS.contains(&id));
            assert!(!SoundId::STINGERS.contains(&id));
        }
    }

    #[test]
    fn movement_motifs_deterministic_distinct_and_unclipped() {
        let a = SynthBank::build();
        let b = SynthBank::build();
        // Hash every *other* sound first: a motif must not duplicate an existing
        // recipe's PCM.
        let mut all: std::collections::HashSet<u64> = SoundId::ALL
            .into_iter()
            .filter(|id| !SoundId::MOVEMENT_MOTIFS.contains(id))
            .map(|id| fnv1a_i16(&a.get(id).to_i16()))
            .collect();
        for id in SoundId::MOVEMENT_MOTIFS {
            let s = a.get(id);
            let ha = fnv1a_i16(&s.to_i16());
            assert_eq!(
                ha,
                fnv1a_i16(&b.get(id).to_i16()),
                "{} not deterministic across builds",
                id.name()
            );
            assert_eq!(s.frames, b.get(id).frames);
            assert!(all.insert(ha), "{} duplicated another sound", id.name());
            let (peak, _) = peak_of(&s.frames);
            assert!(
                peak > 0.05,
                "{} peaks at only {peak} — inaudible",
                id.name()
            );
            assert!(
                peak < 0.99,
                "{} peaks at {peak} — clipping/clamped",
                id.name()
            );
            assert!(rms_of(&s.frames) > 0.005, "{} is silent", id.name());
        }
    }

    #[test]
    fn movement_motif_shapes_match_their_design() {
        let bank = SynthBank::build();

        // The heartbeat thumps are transients (peak in the opening fifth) and
        // "dub" is the quieter, lower answer to "lub".
        let mut heart_peaks = Vec::new();
        for id in [SoundId::HeartLub, SoundId::HeartDub] {
            let s = bank.get(id);
            let (peak, at) = peak_of(&s.frames);
            assert!(
                at < s.frame_count() / 5,
                "{} peaks at frame {at} of {} — not a thump",
                id.name(),
                s.frame_count()
            );
            heart_peaks.push(peak);
        }
        assert!(
            heart_peaks[1] < heart_peaks[0],
            "heart_dub ({}) must be softer than heart_lub ({})",
            heart_peaks[1],
            heart_peaks[0]
        );

        // The gaze pulse and the river wash are swells, not hits: the peak sits
        // past the opening and the first 20 ms is far quieter than the body.
        for id in [SoundId::GazePulse, SoundId::RiverWash] {
            let s = bank.get(id);
            let (peak, at) = peak_of(&s.frames);
            assert!(
                at > s.frame_count() / 10,
                "{} peaks at frame {at} of {} — that is a transient, not a swell",
                id.name(),
                s.frame_count()
            );
            let head = (0.020 * SAMPLE_RATE as f32) as usize;
            let head_peak = peak_of(&s.frames[..head]).0;
            assert!(
                head_peak < 0.30 * peak,
                "{} opens at {head_peak} vs peak {peak} — attack is too abrupt",
                id.name()
            );
        }

        // The river wash is the long one (a bed-like swell, > 2 s); the pulse and
        // the thumps stay short enough not to smear across their beat.
        assert!(bank.get(SoundId::RiverWash).frame_count() > 2 * SAMPLE_RATE as usize);
        for id in [SoundId::GazePulse, SoundId::HeartLub, SoundId::HeartDub] {
            let n = bank.get(id).frame_count();
            assert!(
                n < (0.8 * SAMPLE_RATE as f32) as usize,
                "{} is {n} frames — too long for a beat-level motif",
                id.name()
            );
        }
    }

    #[test]
    fn from_name_round_trips_every_sound() {
        // The data-driven hook a trigger→sound table relies on.
        for id in SoundId::ALL {
            assert_eq!(SoundId::from_name(id.name()), Some(id), "{}", id.name());
        }
        assert_eq!(
            SoundId::from_name("stinger_notice"),
            Some(SoundId::StingerNotice)
        );
        assert_eq!(
            SoundId::from_name("stinger_memory"),
            Some(SoundId::StingerMemory)
        );
        assert_eq!(
            SoundId::from_name("stinger_corrupt"),
            Some(SoundId::StingerCorrupt)
        );
        assert_eq!(SoundId::from_name("no_such_sound"), None);
        assert_eq!(SoundId::from_name(""), None);
    }

    // ───────────────────────── the sound grammar ─────────────────────────────

    /// Gain-weighted glide direction of a recipe's partials: > 0 rising,
    /// < 0 falling, ≈ 0 flat. Log-ratio so an octave up and an octave down are
    /// symmetric.
    fn glide_of(id: SoundId) -> f32 {
        let rc = &RECIPES[id.index()];
        let mut acc = 0.0f32;
        let mut w = 0.0f32;
        for pt in rc.partials.iter().take(rc.partial_count) {
            if pt.freq0 > 0.0 && pt.freq1 > 0.0 {
                acc += pt.gain * (pt.freq1 / pt.freq0).log2();
                w += pt.gain;
            }
        }
        if w > 0.0 {
            acc / w
        } else {
            0.0
        }
    }

    /// Is the ratio of two frequencies a minor-second clash (or a sub-3 %
    /// detune that beats)? Octave-folded into `[1, 2)`.
    fn dissonant_pair(a: f32, b: f32) -> bool {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let mut r = hi / lo;
        while r >= 2.0 {
            r /= 2.0;
        }
        let near = |x: f32| (r - x).abs() / x < 0.015;
        (r > 1.0005 && r < 1.03) || near(16.0 / 15.0) || near(15.0 / 8.0) || near(45.0 / 32.0)
    }

    /// Every partial pair is a simple consonance (unison/octave, fifth, fourth,
    /// major third/sixth) — octave-folded ratios within 1 %.
    fn consonant_pair(a: f32, b: f32) -> bool {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let mut r = hi / lo;
        while r >= 2.0 - 1e-3 {
            r /= 2.0;
        }
        [1.0f32, 3.0 / 2.0, 4.0 / 3.0, 5.0 / 4.0, 5.0 / 3.0]
            .iter()
            .any(|&x| (r - x).abs() / x < 0.01)
    }

    fn end_freqs(id: SoundId) -> Vec<f32> {
        let rc = &RECIPES[id.index()];
        rc.partials
            .iter()
            .take(rc.partial_count)
            .filter(|p| p.gain > 0.0 && p.freq1 > 0.0)
            .map(|p| p.freq1)
            .collect()
    }

    #[test]
    fn grammar_rising_means_progress_falling_means_regress() {
        for id in [
            SoundId::SwitchOn,
            SoundId::NodeLit,
            SoundId::AimTick,
            SoundId::NoteRead,
            SoundId::PortalWarp,
            SoundId::StoreyUp,
        ] {
            assert!(
                glide_of(id) > 0.05,
                "{} must rise (glide {})",
                id.name(),
                glide_of(id)
            );
        }
        for id in [
            SoundId::SwitchOff,
            SoundId::NodeUnlit,
            SoundId::RefusePower,
            SoundId::RefuseCycle,
            SoundId::StoreyDown,
        ] {
            assert!(
                glide_of(id) < -0.05,
                "{} must fall (glide {})",
                id.name(),
                glide_of(id)
            );
        }
        // "Hit nothing" is neutral: no pitch gesture at all.
        assert!(glide_of(SoundId::InteractEmpty).abs() < 0.02);
    }

    #[test]
    fn grammar_consonant_means_correct_dissonant_means_blocked() {
        for id in [
            SoundId::SwitchOn,
            SoundId::NodeLit,
            SoundId::CircuitResolve,
            SoundId::FirstLight,
        ] {
            let f = end_freqs(id);
            for i in 0..f.len() {
                for j in (i + 1)..f.len() {
                    assert!(
                        consonant_pair(f[i], f[j]),
                        "{}: {} / {} is not a consonance",
                        id.name(),
                        f[i],
                        f[j]
                    );
                }
            }
        }
        for id in [
            SoundId::RefusePower,
            SoundId::RefuseCycle,
            SoundId::NodeUnlit,
        ] {
            let f = end_freqs(id);
            let clash =
                (0..f.len()).any(|i| ((i + 1)..f.len()).any(|j| dissonant_pair(f[i], f[j])));
            assert!(clash, "{} carries no dissonant pair: {f:?}", id.name());
        }
    }

    /// Autocorrelation pitch estimate (Hz) of a window, searching 80..2000 Hz.
    fn pitch_at(frames: &[f32], center: usize) -> f32 {
        let half = 1024usize;
        let start = center.saturating_sub(half);
        let end = (center + half).min(frames.len());
        let w = &frames[start..end];
        let sr = SAMPLE_RATE as f32;
        let (min_lag, max_lag) = ((sr / 2000.0) as usize, (sr / 80.0) as usize);
        let energy: f32 = w.iter().map(|v| v * v).sum();
        let ac = |lag: usize| -> f32 {
            let mut c = 0.0f32;
            for i in 0..(w.len() - lag) {
                c += w[i] * w[i + lag];
            }
            c / energy.max(1e-9)
        };
        // Skip the zero-lag lobe: only consider lags after the correlation
        // first goes negative, then take the strongest peak.
        let top = max_lag.min(w.len() / 2);
        let mut lag = min_lag;
        while lag < top && ac(lag) > 0.0 {
            lag += 1;
        }
        let mut best = (f32::MIN, lag);
        for l in lag..top {
            let c = ac(l);
            if c > best.0 {
                best = (c, l);
            }
        }
        sr / best.1 as f32
    }

    #[test]
    fn grammar_pitch_direction_is_audible_in_the_pcm() {
        // Not just the recipe data: the rendered PCM itself goes up / down.
        let bank = SynthBank::build();
        for (id, rising) in [
            (SoundId::SwitchOn, true),
            (SoundId::SwitchOff, false),
            (SoundId::StoreyUp, true),
            (SoundId::StoreyDown, false),
            (SoundId::NodeUnlit, false),
        ] {
            let s = &bank.get(id).frames;
            let a = pitch_at(s, s.len() / 4);
            let b = pitch_at(s, s.len() * 3 / 4);
            if rising {
                assert!(b > a * 1.05, "{}: {a} Hz → {b} Hz is not rising", id.name());
            } else {
                assert!(
                    b < a * 0.95,
                    "{}: {a} Hz → {b} Hz is not falling",
                    id.name()
                );
            }
        }
    }

    #[test]
    fn on_and_off_never_sound_alike() {
        let bank = SynthBank::build();
        let on = &bank.get(SoundId::SwitchOn).frames;
        let off = &bank.get(SoundId::SwitchOff).frames;
        assert_ne!(
            fnv1a_i16(&bank.get(SoundId::SwitchOn).to_i16()),
            fnv1a_i16(&bank.get(SoundId::SwitchOff).to_i16())
        );
        // And they are not just a time-reversal of the same energy: OFF is
        // quieter (closing is not an event to celebrate).
        assert!(
            rms_of(off) < rms_of(on),
            "switch_off must sit under switch_on"
        );
    }

    #[test]
    fn grammar_cues_stay_out_of_the_generative_palettes() {
        let bank = SynthBank::build();
        assert_eq!(&SoundId::ALL[39..], &SoundId::GRAMMAR[..]);
        for id in SoundId::GRAMMAR {
            assert!(!SoundId::GLITCH_FAMILY.contains(&id), "{}", id.name());
            assert!(!SoundId::TONAL_PALETTE.contains(&id), "{}", id.name());
            assert!(!SoundId::DRONE_MOODS.contains(&id), "{}", id.name());
            let s = bank.get(id);
            let (peak, _) = peak_of(&s.frames);
            assert!(peak > 0.05 && peak < 0.99, "{} peak {peak}", id.name());
        }
        // Distinct PCM from every sound that existed before them.
        let mut seen: std::collections::HashSet<u64> = SoundId::ALL[..39]
            .iter()
            .map(|&id| fnv1a_i16(&bank.get(id).to_i16()))
            .collect();
        for id in SoundId::GRAMMAR {
            assert!(
                seen.insert(fnv1a_i16(&bank.get(id).to_i16())),
                "{} duplicates",
                id.name()
            );
        }
    }

    #[test]
    fn noise_is_deterministic_and_bounded() {
        let mut a = Lcg(0x1357);
        let mut b = Lcg(0x1357);
        for _ in 0..1000 {
            let x = noise(&mut a);
            let y = noise(&mut b);
            assert_eq!(x, y);
            assert!((-1.0..1.0).contains(&x));
        }
    }
}
