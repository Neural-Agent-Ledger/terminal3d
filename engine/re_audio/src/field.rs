//! Pure, device-free "sound fields" mixing.
//!
//! `combine()` folds a slice of weighted [`FieldInput`]s — one per audible
//! field, already *selected* (top-3 + hysteresis + always-on room fields)
//! and *smoothed* (attack/release) by the host game — into a [`FieldMix`]
//! describing the target state of the field insert chain on every bus (bus
//! topology, chain slots, and mixing rules).
//!
//! This module has **zero kira dependency**: it is plain data in, plain
//! data out, and is usable from any host (game, editor, or a standalone
//! tool) with no audio device. `Player::set_fields` (in `playback.rs`) is
//! what turns a `FieldMix` into kira parameter tweens on the real graph.
//!
//! # Interpretation notes (this file's job is `combine`, not selection)
//!
//! Selecting which fields are audible each frame requires per-frame state —
//! an attack/release envelope per candidate field, and hysteresis on the
//! top-3 cut ("a field must beat the third-place field by 10% to replace
//! it"). `combine`'s frozen signature (`&[FieldInput], &FieldSettings) ->
//! FieldMix`) is stateless, so it cannot own that memory: **selection and
//! envelope-smoothing are the host game's job**; `combine` maps each input to
//! its archetype recipe, combines per-kind across simultaneous fields,
//! resolves the few winner-take-all parameters, and applies the floors/caps.
//! `combine` is also where the "room-shaped fields are always in addition to
//! the three" exception is trivially satisfied: the host just includes them
//! in the slice it passes.
//!
//! # Design choices for the custom effect chain
//!
//! The five custom effects in `field_fx/` use kira's own `Parameter<T>`
//! (tween-based smoothing — the same mechanism every kira-native effect in
//! the chain already uses, so the whole chain smooths uniformly) and this
//! crate's existing `synth::Lcg` deterministic generator (the same noise
//! family `dsp.rs` and `soundscape.rs` already standardize on), rather than a
//! second smoothing mechanism or a second noise primitive. Both give the
//! properties that matter — click-free, deterministic given a seed, no
//! audio-thread allocation.

use crate::field_fx::glitch_stutter::HoldMode;

/// Bus index: the ambience/drone bed track.
pub const AMB: usize = 0;
/// Bus index: the drone-music/harmony track.
pub const MUS: usize = 1;
/// Bus index: the one-shot sound-effects track.
pub const SFX: usize = 2;
/// Bus index: the UI/cue track (reduced chain — no wet effect slots).
pub const CUE: usize = 3;
/// Bus index: the positional/spatial-voice track.
pub const POS: usize = 4;
/// Number of group tracks a [`FieldMix`] carries a [`BusFx`] for.
pub const BUS_COUNT: usize = 5;
const BUS_IDS: [usize; BUS_COUNT] = [AMB, MUS, SFX, CUE, POS];

/// One of the 15 lore-bound field archetypes. Each names a distinct
/// atmospheric treatment (`apply_archetype` holds its exact bus recipe);
/// together they are the vocabulary [`combine`] mixes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Archetype {
    /// The LETHE river's muffling wash: closes the low-pass, softens highs,
    /// narrows the stereo image, and adds a little bit-rot dropout.
    LetheWash,
    /// MNEMOS holding a cycle: a growing delay/echo tail, freezing to a
    /// sustained loop while the puzzle cycle stays held.
    MnemosHold,
    /// Data corruption texture: crushes bit depth and sample rate, with a
    /// touch of distortion and dropout.
    BitRot,
    /// A memory replaying itself: wow/flutter pitch wobble plus an echo tail,
    /// clearer or smearier depending on the memory's `clarity`.
    FalseEcho,
    /// Something watching: narrows stereo width and ducks volume across
    /// every bus but CUE, with a presence boost on SFX.
    WatcherGaze,
    /// A gap in the world where a voice should be: ring-modulates AMB/MUS
    /// toward a harmony-locked carrier.
    PhantomHollow,
    /// A lure pulling the player off-course: tilts EQ, adds light distortion,
    /// and a slow wow wobble.
    LureWobble,
    /// The exit door drawing near: detunes the loop pitch downward and carves
    /// a hollow notch in AMB.
    ExitDetune,
    /// An electrified hazard nearby: a harmony-locked bell boost plus a
    /// rhythmic volume pump on POS/SFX.
    LiveWire,
    /// Structural strain building toward release: a pumping volume LFO on
    /// AMB/MUS whose rate steps with the strain ratio.
    ThresholdStrain,
    /// The power grid near its cap: darkens the low-pass and, once at cap,
    /// ducks volume — the steady target a per-frame flicker sits on top of.
    Brownout,
    /// A tail sound trailing off in the distance: a tick-synced stutter on
    /// SFX/AMB whose subdivision follows distance.
    TailF,
    /// A tightening focus that shuts out the periphery: narrows width and
    /// lifts the high-pass on every bus but CUE, brightens SFX, and ducks AMB.
    RecognitionNarrow,
    /// Warm ambient steam/heat: gentle EQ tilt, a little wow, and extra send
    /// into the shared reverb.
    SteamWarmth,
    /// Sky/void ambience with several treatments selected by `variant`:
    /// subtle data-rain crush, an aurora wow shimmer, a watcher's narrowing
    /// hush, or (default) a distant, muffled void.
    SkyHush,
}

impl Archetype {
    /// Every archetype, in a stable, arbitrary order used for iteration and tests.
    pub const ALL: [Archetype; 15] = [
        Archetype::LetheWash,
        Archetype::MnemosHold,
        Archetype::BitRot,
        Archetype::FalseEcho,
        Archetype::WatcherGaze,
        Archetype::PhantomHollow,
        Archetype::LureWobble,
        Archetype::ExitDetune,
        Archetype::LiveWire,
        Archetype::ThresholdStrain,
        Archetype::Brownout,
        Archetype::TailF,
        Archetype::RecognitionNarrow,
        Archetype::SteamWarmth,
        Archetype::SkyHush,
    ];

    /// The archetype's stable lowercase `snake_case` name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Archetype::LetheWash => "lethe_wash",
            Archetype::MnemosHold => "mnemos_hold",
            Archetype::BitRot => "bit_rot",
            Archetype::FalseEcho => "false_echo",
            Archetype::WatcherGaze => "watcher_gaze",
            Archetype::PhantomHollow => "phantom_hollow",
            Archetype::LureWobble => "lure_wobble",
            Archetype::ExitDetune => "exit_detune",
            Archetype::LiveWire => "live_wire",
            Archetype::ThresholdStrain => "threshold_strain",
            Archetype::Brownout => "brownout",
            Archetype::TailF => "tail_f",
            Archetype::RecognitionNarrow => "recognition_narrow",
            Archetype::SteamWarmth => "steam_warmth",
            Archetype::SkyHush => "sky_hush",
        }
    }

    /// Parses [`Archetype::name`]'s output back into an `Archetype`. `None` for
    /// any other string.
    #[must_use]
    pub fn from_name(s: &str) -> Option<Archetype> {
        Archetype::ALL.into_iter().find(|a| a.name() == s)
    }

    /// Default priority, used as the tie-breaker score (`priority * weight`)
    /// for the handful of parameters that must pick one winner across several
    /// simultaneous fields rather than blend. `ExitDetune` is highest (9): the
    /// exit door is the strongest signal. `MnemosHold` is 9 only while its
    /// field's `variant` signals `cycle_held` (see [`apply_archetype`]); its
    /// resting priority is 7, grouped with the other puzzle-state archetypes.
    #[must_use]
    pub fn default_priority(self) -> u8 {
        match self {
            Archetype::ExitDetune => 9,
            Archetype::RecognitionNarrow
            | Archetype::ThresholdStrain
            | Archetype::PhantomHollow
            | Archetype::MnemosHold => 7,
            Archetype::BitRot
            | Archetype::FalseEcho
            | Archetype::WatcherGaze
            | Archetype::LureWobble
            | Archetype::Brownout => 5,
            Archetype::LetheWash | Archetype::SteamWarmth | Archetype::LiveWire => 3,
            Archetype::TailF | Archetype::SkyHush => 1,
        }
    }
}

/// One audible field's contribution this frame. The host game produces these
/// after selecting the top candidates and envelope-smoothing their weights;
/// `combine` consumes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldInput {
    /// Which archetype recipe this input drives.
    pub archetype: Archetype,
    /// The smoothed shape/gate/drive weight `w_i` in `[0, 1]`.
    pub weight: f32,
    /// The authored `depth =` in `[0, 1]`.
    pub depth: f32,
    /// `priority =` (defaults to [`Archetype::default_priority`] upstream).
    pub priority: u8,
    /// Archetype-specific extra scalar: clarity (`FalseEcho`), strain ratio
    /// (`ThresholdStrain`), budget (`Brownout`), distance in cells (`TailF`),
    /// harmony root Hz (`PhantomHollow`/`LiveWire`), or unused (0.0) for
    /// archetypes with no such input.
    pub aux: f32,
    /// Archetype-specific discrete selector: `false_echo`'s `variant=memory`
    /// (1), `mnemos_hold`'s `cycle_held` latch (1), `sky_hush`'s variant
    /// (0=void 1=datarain 2=aurora 3=watcher 4=storm), else 0.
    pub variant: u8,
    /// Deterministic seed for this field's noise-driven slots (Crush
    /// dropout, Brownout's flicker).
    pub seed: u64,
    /// World-space bearing for effects that need a direction (currently
    /// unused by `combine`'s pure math; carried through for the host game's/
    /// `playback.rs`'s spatial bookkeeping, e.g. `FalseEcho`'s mirror voice).
    pub bearing: Option<[f32; 3]>,
}

/// Global accessibility + intensity settings for the whole field system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldSettings {
    /// `sound_fields`: Off/Low/Medium/Full = 0.0/0.33/0.66/1.0. `0.0` makes
    /// `combine` return [`FieldMix::NEUTRAL`] bit-exact.
    pub intensity: f32,
    /// `sound_fields_reduced`: an independent accessibility setting, default
    /// `false` — not tied to the motion-sensitivity/glitch setting.
    pub reduced: bool,
}

impl Default for FieldSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            reduced: false,
        }
    }
}

/// The field insert chain's target state for one group track (the chain's
/// dozen effect slots, minus the fixed-at-build `DL` time). All fields are
/// expressed as **absolute target values** ready to tween a kira handle to
/// directly — not deltas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BusFx {
    /// `LS`: low-shelf gain, dB, `[-18, 6]` (`combine` clamps to the tighter
    /// `[-12, 6]` shelf cap at finalize time).
    pub ls_db: f32,
    /// `BL`: bell center frequency, Hz.
    pub bl_freq_hz: f32,
    /// `BL`: bell Q.
    pub bl_q: f32,
    /// `BL`: bell gain, dB, `[-12, 6]`.
    pub bl_gain_db: f32,
    /// `HS`: high-shelf gain, dB, `[-18, 6]`.
    pub hs_db: f32,
    /// `LP`: cutoff, Hz. Floor 350Hz (general buses) or the CUE-specific
    /// floor; `combine` enforces both.
    pub lp_cutoff_hz: f32,
    /// `HP`: cutoff, Hz, `[20, 1200]`.
    pub hp_cutoff_hz: f32,
    /// `CR`: bit depth, `[6, 16]` (CUE floor 12).
    pub crush_bits: f32,
    /// `CR`: sample-and-hold length, samples, `[1, 8]`.
    pub crush_hold: f32,
    /// `CR`: dropout probability per S&H capture, `[0, 0.25]`. Always 0 on
    /// CUE.
    pub crush_dropout: f32,
    /// `CR`: dry/wet mix, part of the chain's `W_max` budget.
    pub crush_mix: f32,
    /// `RM`: carrier, Hz.
    pub ringmod_carrier_hz: f32,
    /// `RM`: dry/wet mix, `W_max` budget.
    pub ringmod_mix: f32,
    /// `WW`: fixed base delay, ms.
    pub wow_base_ms: f32,
    /// `WW`: peak modulation depth around `base`, ms.
    pub wow_depth_ms: f32,
    /// `WW`: modulation rate, Hz.
    pub wow_rate_hz: f32,
    /// `WW`: dry/wet mix, `W_max` budget.
    pub wow_mix: f32,
    /// `HD`: mode.
    pub hold_mode: HoldMode,
    /// `HD`: capture window, ms.
    pub hold_window_ms: f32,
    /// `HD`: dry/wet mix, `[0, 0.7]`, `W_max` budget.
    pub hold_mix: f32,
    /// `DS`: `SoftClip` drive, dB, `[0, 18]`.
    pub dist_drive_db: f32,
    /// `DS`: dry/wet mix, `[0, 0.5]`, `W_max` budget.
    pub dist_mix: f32,
    /// `WD`: stereo width, `[0.2, 1.3]` (the floor is raised further in
    /// reduced-accessibility mode and on the CUE bus).
    pub width: f32,
    /// `DL`: feedback, dB, `<= -3`.
    pub delay_feedback_db: f32,
    /// `DL`: dry/wet mix, `[0, 0.5]`, `W_max` budget.
    pub delay_mix: f32,
    /// `VOL`: static duck, dB, `>= -9` (CUE `>= -3`).
    pub vol_duck_db: f32,
    /// `VOL`: LFO pump depth, dB, `<= 3`.
    pub pump_depth_db: f32,
    /// `VOL`: LFO pump rate, Hz.
    pub pump_rate_hz: f32,
    /// `SEND`: level into `SPACE`, dB, `<= -3`.
    pub send_db: f32,
}

impl BusFx {
    /// The bypass state: every effect at its identity value, bit-exact
    /// passthrough on every slot.
    pub const NEUTRAL: BusFx = BusFx {
        ls_db: 0.0,
        bl_freq_hz: 1000.0,
        bl_q: 1.0,
        bl_gain_db: 0.0,
        hs_db: 0.0,
        lp_cutoff_hz: OPEN_LP_HZ,
        hp_cutoff_hz: OPEN_HP_HZ,
        crush_bits: 16.0,
        crush_hold: 1.0,
        crush_dropout: 0.0,
        crush_mix: 0.0,
        ringmod_carrier_hz: 200.0,
        ringmod_mix: 0.0,
        wow_base_ms: 8.0,
        wow_depth_ms: 0.0,
        wow_rate_hz: 0.6,
        wow_mix: 0.0,
        hold_mode: HoldMode::Off,
        hold_window_ms: 60.0,
        hold_mix: 0.0,
        dist_drive_db: 0.0,
        dist_mix: 0.0,
        width: 1.0,
        delay_feedback_db: OFF_DB,
        delay_mix: 0.0,
        vol_duck_db: 0.0,
        pump_depth_db: 0.0,
        pump_rate_hz: 2.0,
        send_db: OFF_DB,
    };
}

/// The shared `SPACE` send: `Hold` (freeze-capable) in front of a reverb,
/// fed by a per-bus send level ([`BusFx::send_db`]) rather than being a
/// per-bus insert itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceFx {
    /// `Hold` mode on the shared send (see [`BusFx::hold_mode`]).
    pub hold_mode: HoldMode,
    /// `Hold` capture window, ms.
    pub hold_window_ms: f32,
    /// `Hold` dry/wet mix.
    pub hold_mix: f32,
    /// Reverb feedback (decay length), `<= REVERB_FEEDBACK_CEIL`.
    pub reverb_feedback: f32,
    /// Reverb high-frequency damping, `[0, 1]`.
    pub reverb_damping: f32,
}

impl SpaceFx {
    /// The bypass state: `Hold` off, reverb at its resting decay/damping.
    pub const NEUTRAL: SpaceFx = SpaceFx {
        hold_mode: HoldMode::Off,
        hold_window_ms: 60.0,
        hold_mix: 0.0,
        reverb_feedback: 0.85,
        reverb_damping: 0.5,
    };
}

/// The full target state produced by [`combine`]: one [`BusFx`] per group
/// track (`AMB MUS SFX CUE POS`, indexed by [`AMB`]/[`MUS`]/.../[`POS`]),
/// the shared [`SpaceFx`] send, and the global loop detune (`RATE`, driven
/// by `ExitDetune`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldMix {
    /// Per-group-track effect state, indexed by [`AMB`]/[`MUS`]/[`SFX`]/[`CUE`]/[`POS`].
    pub buses: [BusFx; BUS_COUNT],
    /// The shared `SPACE` send state.
    pub space: SpaceFx,
    /// `RATE`: bed/music `playback_rate` detune, cents, `[-35, 35]`
    /// (`[-15, 15]` in reduced mode).
    pub loop_rate_cents: f32,
}

impl FieldMix {
    /// The bypass state: every bus and the shared send at their neutral
    /// values, zero loop detune.
    pub const NEUTRAL: FieldMix = FieldMix {
        buses: [BusFx::NEUTRAL; BUS_COUNT],
        space: SpaceFx::NEUTRAL,
        loop_rate_cents: 0.0,
    };
}

const OPEN_LP_HZ: f32 = 20_000.0;
const OPEN_HP_HZ: f32 = 20.0;
/// A send/feedback level low enough to be inaudible; used as the "off" rest
/// value for `SEND` and `DL feedback`, both specified in dB with no natural
/// zero.
const OFF_DB: f32 = -60.0;

const LP_FLOOR_HZ: f32 = 350.0;
const LP_FLOOR_CUE_HZ: f32 = 1_500.0;
const HP_CEIL_HZ: f32 = 1_200.0;
const SHELF_DB_RANGE: (f32, f32) = (-18.0, 6.0);
const BELL_DB_RANGE: (f32, f32) = (-12.0, 6.0);
const CRUSH_BITS_FLOOR: f32 = 6.0;
const CRUSH_BITS_FLOOR_CUE: f32 = 12.0;
const CRUSH_HOLD_CEIL: f32 = 8.0;
const DROPOUT_CEIL: f32 = 0.25;
const WIDTH_FLOOR_FULL: f32 = 0.3;
const WIDTH_FLOOR_REDUCED: f32 = 0.7;
const WIDTH_FLOOR_CUE: f32 = 0.6;
const RATE_CEIL_CENTS_FULL: f32 = 35.0;
const RATE_CEIL_CENTS_REDUCED: f32 = 15.0;
const VOL_DUCK_FLOOR_DB: f32 = -9.0;
const VOL_DUCK_FLOOR_CUE_DB: f32 = -3.0;
const PUMP_DEPTH_CEIL_DB: f32 = 3.0;
const SEND_CEIL_DB: f32 = -3.0;
const DL_FEEDBACK_CEIL_DB: f32 = -3.0;
const REVERB_FEEDBACK_CEIL: f32 = 0.97;
const W_MAX_FULL: f32 = 0.60;
const W_MAX_REDUCED: f32 = 0.35;

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn lerp_log2(a_hz: f32, b_hz: f32, t: f32) -> f32 {
    2f32.powf(lerp(a_hz.max(1.0).log2(), b_hz.max(1.0).log2(), t))
}

/// Per-bus accumulator. Mirrors [`BusFx`] but every field tracks *how* it is
/// combined across multiple simultaneous fields: sums for dB-additive
/// quantities, min/max (log2 for frequencies) for "most extreme wins"
/// quantities, probabilistic combine (`1 - Π(1-p)`) for wet mixes, weighted
/// average for anything without a more specific combine rule (frequencies/
/// Qs/rates), and true winner-take-all (highest `priority * weight`) for the
/// three parameters that must pick one value rather than blend: `HD` mode,
/// `RM` carrier, `WW` rate.
#[derive(Clone, Copy)]
struct BusAccum {
    ls_db: f32,
    bl_gain_db: f32,
    bl_freq_wsum: f32,
    bl_q_wsum: f32,
    bl_w: f32,
    hs_db: f32,
    lp_cutoff_hz: f32,
    hp_cutoff_hz: f32,
    crush_bits: f32,
    crush_hold: f32,
    crush_dropout_survival: f32,
    crush_mix_survival: f32,
    rm_mix_survival: f32,
    rm_carrier_score: f32,
    rm_carrier_hz: f32,
    ww_mix_survival: f32,
    ww_base_wsum: f32,
    ww_depth_wsum: f32,
    ww_w: f32,
    ww_rate_score: f32,
    ww_rate_hz: f32,
    hd_mix_survival: f32,
    hd_mode_score: f32,
    hd_mode: HoldMode,
    hd_window_wsum: f32,
    hd_w: f32,
    ds_mix_survival: f32,
    ds_drive_db: f32,
    width: f32,
    dl_feedback_db: f32,
    dl_mix_survival: f32,
    vol_duck_db: f32,
    pump_depth_db: f32,
    pump_rate_wsum: f32,
    pump_w: f32,
    send_db: f32,
}

impl BusAccum {
    fn new() -> Self {
        Self {
            ls_db: 0.0,
            bl_gain_db: 0.0,
            bl_freq_wsum: 0.0,
            bl_q_wsum: 0.0,
            bl_w: 0.0,
            hs_db: 0.0,
            lp_cutoff_hz: OPEN_LP_HZ,
            hp_cutoff_hz: OPEN_HP_HZ,
            crush_bits: 16.0,
            crush_hold: 1.0,
            crush_dropout_survival: 1.0,
            crush_mix_survival: 1.0,
            rm_mix_survival: 1.0,
            rm_carrier_score: f32::MIN,
            rm_carrier_hz: BusFx::NEUTRAL.ringmod_carrier_hz,
            ww_mix_survival: 1.0,
            ww_base_wsum: 0.0,
            ww_depth_wsum: 0.0,
            ww_w: 0.0,
            ww_rate_score: f32::MIN,
            ww_rate_hz: BusFx::NEUTRAL.wow_rate_hz,
            hd_mix_survival: 1.0,
            hd_mode_score: f32::MIN,
            hd_mode: HoldMode::Off,
            hd_window_wsum: 0.0,
            hd_w: 0.0,
            ds_mix_survival: 1.0,
            ds_drive_db: 0.0,
            width: 1.0,
            dl_feedback_db: OFF_DB,
            dl_mix_survival: 1.0,
            vol_duck_db: 0.0,
            pump_depth_db: 0.0,
            pump_rate_wsum: 0.0,
            pump_w: 0.0,
            send_db: OFF_DB,
        }
    }

    // -- generic combine helpers, called by archetype recipes below --

    fn add_db(acc: &mut f32, target_db: f32, g: f32) {
        *acc += target_db * g;
    }
    fn min_hz(acc: &mut f32, target_hz: f32, g: f32, open_hz: f32) {
        *acc = acc.min(lerp_log2(open_hz, target_hz, g));
    }
    fn max_hz(acc: &mut f32, target_hz: f32, g: f32, open_hz: f32) {
        *acc = acc.max(lerp_log2(open_hz, target_hz, g));
    }
    fn min_lin(acc: &mut f32, target: f32, g: f32, open: f32) {
        *acc = acc.min(lerp(open, target, g));
    }
    fn max_lin(acc: &mut f32, target: f32, g: f32, open: f32) {
        *acc = acc.max(lerp(open, target, g));
    }
    fn probabilistic(survival: &mut f32, target_mix: f32, g: f32) {
        let p = (target_mix * g).clamp(0.0, 1.0);
        *survival *= 1.0 - p;
    }
    fn weighted(wsum: &mut f32, w: &mut f32, target: f32, g: f32) {
        *wsum += target * g;
        *w += g;
    }
    fn categorical<T: Copy>(score_slot: &mut f32, value_slot: &mut T, score: f32, value: T) {
        if score > *score_slot {
            *score_slot = score;
            *value_slot = value;
        }
    }

    fn finalize(self, bus: usize, reduced: bool) -> BusFx {
        let is_cue = bus == CUE;
        let width_floor = if is_cue {
            WIDTH_FLOOR_CUE
        } else if reduced {
            WIDTH_FLOOR_REDUCED
        } else {
            WIDTH_FLOOR_FULL
        };
        let lp_floor = if is_cue { LP_FLOOR_CUE_HZ } else { LP_FLOOR_HZ };
        let duck_floor = if is_cue {
            VOL_DUCK_FLOOR_CUE_DB
        } else {
            VOL_DUCK_FLOOR_DB
        };
        let crush_bits_floor = if is_cue {
            CRUSH_BITS_FLOOR_CUE
        } else {
            CRUSH_BITS_FLOOR
        };

        let wavg = |wsum: f32, w: f32, neutral: f32| if w > 1e-6 { wsum / w } else { neutral };

        let mut crush_mix = 1.0 - self.crush_mix_survival;
        let mut rm_mix = 1.0 - self.rm_mix_survival;
        let mut ww_mix = 1.0 - self.ww_mix_survival;
        let mut hd_mix = 1.0 - self.hd_mix_survival;
        let mut ds_mix = 1.0 - self.ds_mix_survival;
        let mut dl_mix = 1.0 - self.dl_mix_survival;

        // CUE never carries the wet-effect slots at all — its reduced chain
        // is LP, WD, VOL only.
        if is_cue {
            crush_mix = 0.0;
            rm_mix = 0.0;
            ww_mix = 0.0;
            hd_mix = 0.0;
            ds_mix = 0.0;
            dl_mix = 0.0;
        }

        let w_max = if reduced { W_MAX_REDUCED } else { W_MAX_FULL };
        let wet_sum = crush_mix + rm_mix + ww_mix + hd_mix + ds_mix + dl_mix;
        if wet_sum > w_max && wet_sum > 0.0 {
            let scale = w_max / wet_sum;
            crush_mix *= scale;
            rm_mix *= scale;
            ww_mix *= scale;
            hd_mix *= scale;
            ds_mix *= scale;
            dl_mix *= scale;
        }

        let mut dropout = if is_cue {
            0.0
        } else {
            (1.0 - self.crush_dropout_survival).min(DROPOUT_CEIL)
        };
        if reduced {
            // Reduced-accessibility mode: crush dropout, ring-mod, and
            // hold stutter/freeze are all motion/glitch-sensitive, so they
            // are removed outright rather than merely turned down.
            dropout = 0.0;
            rm_mix = 0.0;
            hd_mix = 0.0;
        }

        BusFx {
            ls_db: self.ls_db.clamp(SHELF_DB_RANGE.0, SHELF_DB_RANGE.1),
            bl_freq_hz: wavg(self.bl_freq_wsum, self.bl_w, BusFx::NEUTRAL.bl_freq_hz),
            bl_q: wavg(self.bl_q_wsum, self.bl_w, BusFx::NEUTRAL.bl_q),
            bl_gain_db: self.bl_gain_db.clamp(BELL_DB_RANGE.0, BELL_DB_RANGE.1),
            hs_db: self.hs_db.clamp(SHELF_DB_RANGE.0, SHELF_DB_RANGE.1),
            lp_cutoff_hz: self.lp_cutoff_hz.max(lp_floor),
            hp_cutoff_hz: self.hp_cutoff_hz.min(HP_CEIL_HZ),
            crush_bits: self.crush_bits.max(crush_bits_floor),
            crush_hold: self.crush_hold.min(CRUSH_HOLD_CEIL),
            crush_dropout: dropout,
            crush_mix,
            ringmod_carrier_hz: self.rm_carrier_hz,
            ringmod_mix: rm_mix,
            wow_base_ms: wavg(self.ww_base_wsum, self.ww_w, BusFx::NEUTRAL.wow_base_ms),
            wow_depth_ms: if reduced {
                // Reduced-accessibility mode dampens wow/flutter depth
                // rather than removing it outright — a subtler wobble.
                wavg(self.ww_depth_wsum, self.ww_w, 0.0) * 0.3
            } else {
                wavg(self.ww_depth_wsum, self.ww_w, 0.0)
            },
            wow_rate_hz: self.ww_rate_hz,
            wow_mix: ww_mix,
            hold_mode: if reduced { HoldMode::Off } else { self.hd_mode },
            hold_window_ms: wavg(
                self.hd_window_wsum,
                self.hd_w,
                BusFx::NEUTRAL.hold_window_ms,
            ),
            hold_mix: hd_mix,
            dist_drive_db: self.ds_drive_db.max(0.0),
            dist_mix: ds_mix,
            width: self.width.max(width_floor),
            delay_feedback_db: self.dl_feedback_db.min(DL_FEEDBACK_CEIL_DB),
            delay_mix: dl_mix,
            vol_duck_db: self.vol_duck_db.max(duck_floor),
            pump_depth_db: self.pump_depth_db.min(PUMP_DEPTH_CEIL_DB),
            pump_rate_hz: wavg(
                self.pump_rate_wsum,
                self.pump_w,
                BusFx::NEUTRAL.pump_rate_hz,
            ),
            send_db: self.send_db.min(SEND_CEIL_DB),
        }
    }
}

struct SpaceAccum {
    hold_mix_survival: f32,
    hold_mode_score: f32,
    hold_mode: HoldMode,
    hold_window_wsum: f32,
    hold_w: f32,
    reverb_feedback: f32,
    reverb_damping_wsum: f32,
    reverb_damping_w: f32,
}

impl SpaceAccum {
    fn new() -> Self {
        Self {
            hold_mix_survival: 1.0,
            hold_mode_score: f32::MIN,
            hold_mode: HoldMode::Off,
            hold_window_wsum: 0.0,
            hold_w: 0.0,
            reverb_feedback: SpaceFx::NEUTRAL.reverb_feedback,
            reverb_damping_wsum: 0.0,
            reverb_damping_w: 0.0,
        }
    }

    fn finalize(self, reduced: bool) -> SpaceFx {
        let wavg = |wsum: f32, w: f32, neutral: f32| if w > 1e-6 { wsum / w } else { neutral };
        let mut hold_mix = 1.0 - self.hold_mix_survival;
        let mut mode = self.hold_mode;
        if reduced {
            hold_mix = 0.0;
            mode = HoldMode::Off;
        }
        SpaceFx {
            hold_mode: mode,
            hold_window_ms: wavg(
                self.hold_window_wsum,
                self.hold_w,
                SpaceFx::NEUTRAL.hold_window_ms,
            ),
            hold_mix,
            reverb_feedback: self.reverb_feedback.min(REVERB_FEEDBACK_CEIL),
            reverb_damping: wavg(
                self.reverb_damping_wsum,
                self.reverb_damping_w,
                SpaceFx::NEUTRAL.reverb_damping,
            ),
        }
    }
}

/// Fold `inputs` into a [`FieldMix`].
///
/// Maps each input to its archetype recipe, combines per-kind across
/// simultaneous fields, resolves the winner-take-all parameters, and applies
/// the floors/caps. Empty input (or `settings.intensity <= 0`) yields exactly
/// [`FieldMix::NEUTRAL`].
#[must_use]
pub fn combine(inputs: &[FieldInput], settings: &FieldSettings) -> FieldMix {
    let intensity = settings.intensity.clamp(0.0, 1.0);
    if inputs.is_empty() || intensity <= 0.0 {
        return FieldMix::NEUTRAL;
    }

    let mut buses: [BusAccum; BUS_COUNT] = [BusAccum::new(); BUS_COUNT];
    let mut space = SpaceAccum::new();
    let mut rate_cents = 0.0f32;

    for input in inputs {
        let g = (input.weight.clamp(0.0, 1.0) * input.depth.clamp(0.0, 1.0) * intensity)
            .clamp(0.0, 1.0);
        if g <= 0.0 {
            continue;
        }
        let score = f32::from(input.priority) * input.weight.clamp(0.0, 1.0);
        apply_archetype(input, g, score, &mut buses, &mut space, &mut rate_cents);
    }

    let rate_ceil = if settings.reduced {
        RATE_CEIL_CENTS_REDUCED
    } else {
        RATE_CEIL_CENTS_FULL
    };

    FieldMix {
        buses: BUS_IDS.map(|b| buses[b].finalize(b, settings.reduced)),
        space: space.finalize(settings.reduced),
        loop_rate_cents: rate_cents.clamp(-rate_ceil, rate_ceil),
    }
}

/// Applies one field's target recipe (one match arm per [`Archetype`]) to the
/// running accumulators, scaled by `g = weight * depth * intensity`.
/// `score = priority * weight` drives the three true winner-take-all slots.
#[allow(clippy::too_many_lines)]
fn apply_archetype(
    input: &FieldInput,
    g: f32,
    score: f32,
    buses: &mut [BusAccum; BUS_COUNT],
    space: &mut SpaceAccum,
    rate_cents: &mut f32,
) {
    match input.archetype {
        Archetype::LetheWash => {
            // F1: AMB MUS SFX POS full chain; CUE gets LP only (floor 2kHz
            // enforced here directly since it's stricter than the general
            // CUE floor of 1.5kHz).
            for &b in &[AMB, MUS, SFX, POS] {
                BusAccum::min_hz(&mut buses[b].lp_cutoff_hz, 700.0, g, OPEN_LP_HZ);
                BusAccum::add_db(&mut buses[b].hs_db, -12.0, g);
                BusAccum::probabilistic(&mut buses[b].crush_dropout_survival, 0.08, g);
                BusAccum::max_lin(&mut buses[b].send_db, -8.0, g, OFF_DB);
                BusAccum::min_lin(&mut buses[b].width, 0.85, g, 1.0);
            }
            BusAccum::min_hz(
                &mut buses[CUE].lp_cutoff_hz,
                2_000.0f32.max(LP_FLOOR_CUE_HZ),
                g,
                OPEN_LP_HZ,
            );
        }
        Archetype::MnemosHold => {
            // F2: SFX POS AMB SPACE. `variant == 1` signals `cycle_held`.
            let held = input.variant == 1;
            for &b in &[SFX, POS, AMB] {
                BusAccum::max_lin(&mut buses[b].dl_feedback_db, -3.0, g, OFF_DB);
                BusAccum::probabilistic(&mut buses[b].dl_mix_survival, 0.35, g);
                if b == AMB {
                    BusAccum::add_db(&mut buses[b].ls_db, 3.0, g);
                }
                let (mode, window) = if held {
                    (HoldMode::Freeze, 500.0)
                } else {
                    (HoldMode::Stutter, 250.0)
                };
                BusAccum::categorical(
                    &mut buses[b].hd_mode_score,
                    &mut buses[b].hd_mode,
                    score,
                    mode,
                );
                BusAccum::weighted(&mut buses[b].hd_window_wsum, &mut buses[b].hd_w, window, g);
                BusAccum::probabilistic(&mut buses[b].hd_mix_survival, 0.4, g);
                if held {
                    BusAccum::max_lin(&mut buses[b].send_db, -6.0, g, OFF_DB);
                }
            }
            let (mode, window) = if held {
                (HoldMode::Freeze, 500.0)
            } else {
                (HoldMode::Stutter, 250.0)
            };
            BusAccum::categorical(
                &mut space.hold_mode_score,
                &mut space.hold_mode,
                score,
                mode,
            );
            BusAccum::weighted(&mut space.hold_window_wsum, &mut space.hold_w, window, g);
            BusAccum::probabilistic(
                &mut space.hold_mix_survival,
                if held { 0.9 } else { 0.4 },
                g,
            );
            let target_feedback = if held { 0.97 } else { 0.85 };
            space.reverb_feedback = space.reverb_feedback.max(lerp(
                SpaceFx::NEUTRAL.reverb_feedback,
                target_feedback,
                g,
            ));
        }
        Archetype::BitRot => {
            // F3: AMB MUS SFX POS; CUE bits never below 12 (enforced by
            // `finalize`'s CUE floor; F3 itself never targets CUE).
            for &b in &[AMB, MUS, SFX, POS] {
                BusAccum::min_lin(&mut buses[b].crush_bits, 6.0, g, 16.0);
                BusAccum::max_lin(&mut buses[b].crush_hold, 6.0, g, 1.0);
                BusAccum::probabilistic(&mut buses[b].crush_dropout_survival, 0.05, g);
                BusAccum::probabilistic(&mut buses[b].crush_mix_survival, 0.5, g);
                BusAccum::max_lin(&mut buses[b].ds_drive_db, 9.0, g, 0.0);
                BusAccum::probabilistic(&mut buses[b].ds_mix_survival, 0.25, g);
                BusAccum::add_db(&mut buses[b].hs_db, 2.0, g);
            }
        }
        Archetype::FalseEcho => {
            // F4: SFX POS AMB (light). `aux` = clarity in [0,1]. The mirror
            // voice one-shot itself is scheduled by `playback.rs` via
            // `MirrorFootstep`, not represented in `FieldMix`.
            let clarity = input.aux.clamp(0.0, 1.0);
            let memory = input.variant == 1;
            if memory {
                // Single fixed bus (AMB only) — no loop needed.
                let b = AMB;
                BusAccum::weighted(&mut buses[b].ww_base_wsum, &mut buses[b].ww_w, 8.0, g);
                BusAccum::weighted(&mut buses[b].ww_depth_wsum, &mut buses[b].ww_w, 3.0, g);
                BusAccum::categorical(
                    &mut buses[b].ww_rate_score,
                    &mut buses[b].ww_rate_hz,
                    score,
                    0.6,
                );
                BusAccum::probabilistic(&mut buses[b].ww_mix_survival, 0.35, g);
                BusAccum::max_lin(&mut buses[b].dl_feedback_db, -12.0, g, OFF_DB);
                BusAccum::probabilistic(&mut buses[b].dl_mix_survival, 0.2, g);
            } else {
                for &b in &[SFX, POS, AMB] {
                    BusAccum::weighted(&mut buses[b].ww_base_wsum, &mut buses[b].ww_w, 12.0, g);
                    BusAccum::weighted(
                        &mut buses[b].ww_depth_wsum,
                        &mut buses[b].ww_w,
                        (1.0 - clarity) * 5.0,
                        g,
                    );
                    BusAccum::categorical(
                        &mut buses[b].ww_rate_score,
                        &mut buses[b].ww_rate_hz,
                        score,
                        0.4,
                    );
                    BusAccum::probabilistic(&mut buses[b].ww_mix_survival, 0.3, g);
                }
                BusAccum::max_lin(&mut buses[SFX].dl_feedback_db, -12.0, g, OFF_DB);
                BusAccum::probabilistic(&mut buses[SFX].dl_mix_survival, 0.15 + 0.25 * clarity, g);
            }
        }
        Archetype::WatcherGaze => {
            // F5: every bus but CUE.
            for &b in &[AMB, MUS, SFX, POS] {
                BusAccum::min_lin(&mut buses[b].width, 0.35, g, 1.0);
                BusAccum::add_db(&mut buses[b].vol_duck_db, -4.0, g);
            }
            BusAccum::weighted(
                &mut buses[SFX].bl_freq_wsum,
                &mut buses[SFX].bl_w,
                2_200.0,
                g,
            );
            BusAccum::weighted(&mut buses[SFX].bl_q_wsum, &mut buses[SFX].bl_w, 1.0, g);
            BusAccum::add_db(&mut buses[SFX].bl_gain_db, 4.0, g);
        }
        Archetype::PhantomHollow => {
            // F6: AMB MUS POS. `aux` = harmony root Hz (falls back to A3
            // when the caller doesn't have one available, e.g. a level with
            // no active harmony voice).
            let root = if input.aux > 0.0 { input.aux } else { 220.0 };
            let carrier = root * (16.0 / 15.0);
            for &b in &[AMB, MUS] {
                BusAccum::categorical(
                    &mut buses[b].rm_carrier_score,
                    &mut buses[b].rm_carrier_hz,
                    score,
                    carrier,
                );
                BusAccum::probabilistic(&mut buses[b].rm_mix_survival, 0.3, g);
            }
            // "Hole in the field": positional gain scaling is a native
            // per-voice concern for `playback.rs`/the host game, not a
            // bus-chain parameter — out of `FieldMix`'s vocabulary by design.
            let _ = POS;
        }
        Archetype::LureWobble => {
            // F7: AMB MUS SFX.
            for &b in &[AMB, MUS, SFX] {
                BusAccum::add_db(&mut buses[b].ls_db, 4.0, g);
                BusAccum::add_db(&mut buses[b].hs_db, -3.0, g);
                BusAccum::max_lin(&mut buses[b].ds_drive_db, 4.0, g, 0.0);
                BusAccum::probabilistic(&mut buses[b].ds_mix_survival, 0.2, g);
                BusAccum::max_lin(&mut buses[b].send_db, -40.0, g, OFF_DB);
            }
            for &b in &[AMB, MUS] {
                BusAccum::weighted(&mut buses[b].ww_base_wsum, &mut buses[b].ww_w, 6.0, g);
                BusAccum::weighted(&mut buses[b].ww_depth_wsum, &mut buses[b].ww_w, 1.8, g);
                BusAccum::categorical(
                    &mut buses[b].ww_rate_score,
                    &mut buses[b].ww_rate_hz,
                    score,
                    0.33,
                );
                BusAccum::probabilistic(&mut buses[b].ww_mix_survival, 1.0, g);
            }
        }
        Archetype::ExitDetune => {
            // ExitDetune: AMB MUS RATE + a hole in the middle. `weight`
            // already encodes `door_ready` inverted (the host game's job);
            // `g` here is the magnitude toward -30 cents.
            *rate_cents += lerp(0.0, -30.0, g);
            BusAccum::weighted(
                &mut buses[AMB].bl_freq_wsum,
                &mut buses[AMB].bl_w,
                1_000.0,
                g,
            );
            BusAccum::weighted(&mut buses[AMB].bl_q_wsum, &mut buses[AMB].bl_w, 2.0, g);
            BusAccum::add_db(&mut buses[AMB].bl_gain_db, -6.0, g);
        }
        Archetype::LiveWire => {
            // F9: POS, plus SFX (light). `aux` = harmony root Hz.
            let root = if input.aux > 0.0 { input.aux } else { 220.0 };
            for &b in &[POS, SFX] {
                BusAccum::max_lin(&mut buses[b].pump_depth_db, 1.5, g, 0.0);
                BusAccum::weighted(&mut buses[b].pump_rate_wsum, &mut buses[b].pump_w, 3.0, g);
                BusAccum::weighted(
                    &mut buses[b].bl_freq_wsum,
                    &mut buses[b].bl_w,
                    root * 4.0,
                    g,
                );
                BusAccum::weighted(&mut buses[b].bl_q_wsum, &mut buses[b].bl_w, 3.0, g);
                BusAccum::add_db(&mut buses[b].bl_gain_db, 2.0, g);
            }
        }
        Archetype::ThresholdStrain => {
            // ThresholdStrain: AMB MUS. `aux` = strain ratio r; the pump
            // rate steps rather than glides (1/2 -> 1.5Hz, 2/3 or 1 -> 3Hz).
            let r = input.aux.clamp(0.0, 1.0);
            let rate = if r <= 0.55 { 1.5 } else { 3.0 };
            for &b in &[AMB, MUS] {
                BusAccum::max_lin(&mut buses[b].pump_depth_db, 3.0, g, 0.0);
                BusAccum::weighted(&mut buses[b].pump_rate_wsum, &mut buses[b].pump_w, rate, g);
            }
        }
        Archetype::Brownout => {
            // Brownout: SFX POS AMB. `aux` = budget in [0,1]; `>= 1.0` is "at
            // cap" (flicker). The flicker itself (a per-frame seeded -2dB
            // sag) is deterministic *display* state computed by
            // `playback.rs`/the host game from `seed`+frame, not a `FieldMix`
            // field — `FieldMix` only carries the steady LP/duck target.
            let budget = input.aux.clamp(0.0, 1.0);
            let at_cap = budget >= 1.0;
            let lp_target = if at_cap {
                2_500.0
            } else {
                lerp(20_000.0, 6_000.0, budget.powi(3))
            };
            for &b in &[SFX, POS, AMB] {
                BusAccum::min_hz(&mut buses[b].lp_cutoff_hz, lp_target, g, OPEN_LP_HZ);
                if at_cap {
                    BusAccum::add_db(&mut buses[b].vol_duck_db, -2.0, g);
                }
            }
        }
        Archetype::TailF => {
            // F12: SFX AMB, tick-synced stutter. `aux` = distance in cells,
            // selecting the repeat subdivision (translated into an
            // approximate rate in Hz at the default 90 BPM / 1.5Hz beat, on
            // top of the fixed clock-driven timing `playback.rs` applies).
            let dist = input.aux.max(0.0);
            let window = if dist < 2.0 { 30.0 } else { 60.0 };
            let subdivisions = if dist > 8.0 {
                0.25 // once per bar (4 beats)
            } else if dist > 4.0 {
                1.0 // once per beat
            } else if dist > 2.0 {
                2.0 // twice per beat
            } else {
                4.0 // four times per beat
            };
            for &b in &[SFX, AMB] {
                BusAccum::categorical(
                    &mut buses[b].hd_mode_score,
                    &mut buses[b].hd_mode,
                    score,
                    HoldMode::Stutter,
                );
                BusAccum::weighted(&mut buses[b].hd_window_wsum, &mut buses[b].hd_w, window, g);
                BusAccum::probabilistic(&mut buses[b].hd_mix_survival, 0.25, g);
                // Encode the subdivision into the pump-rate channel so
                // `playback.rs` can read it back for the clock-tick
                // scheduling; harmless elsewhere since TailF never drives
                // VOL.
                BusAccum::weighted(
                    &mut buses[b].pump_rate_wsum,
                    &mut buses[b].pump_w,
                    subdivisions,
                    g,
                );
            }
        }
        Archetype::RecognitionNarrow => {
            // F13: every bus but CUE.
            for &b in &[AMB, MUS, SFX, POS] {
                BusAccum::min_lin(&mut buses[b].width, 0.3, g, 1.0);
                BusAccum::max_hz(&mut buses[b].hp_cutoff_hz, 160.0, g, OPEN_HP_HZ);
            }
            BusAccum::weighted(
                &mut buses[SFX].bl_freq_wsum,
                &mut buses[SFX].bl_w,
                3_000.0,
                g,
            );
            BusAccum::add_db(&mut buses[SFX].bl_gain_db, 2.0, g);
            BusAccum::add_db(&mut buses[AMB].vol_duck_db, -3.0, g);
        }
        Archetype::SteamWarmth => {
            // F14: AMB SFX POS.
            for &b in &[AMB, SFX, POS] {
                BusAccum::add_db(&mut buses[b].ls_db, 3.0, g);
                BusAccum::add_db(&mut buses[b].hs_db, -2.0, g);
                BusAccum::max_lin(&mut buses[b].send_db, -12.0, g, OFF_DB);
                BusAccum::weighted(&mut buses[b].ww_base_wsum, &mut buses[b].ww_w, 8.0, g);
                BusAccum::weighted(&mut buses[b].ww_depth_wsum, &mut buses[b].ww_w, 0.6, g);
                BusAccum::categorical(
                    &mut buses[b].ww_rate_score,
                    &mut buses[b].ww_rate_hz,
                    score,
                    0.2,
                );
                BusAccum::probabilistic(&mut buses[b].ww_mix_survival, 0.15, g);
            }
            space.reverb_feedback =
                space
                    .reverb_feedback
                    .max(lerp(SpaceFx::NEUTRAL.reverb_feedback, 0.80, g));
            BusAccum::weighted(
                &mut space.reverb_damping_wsum,
                &mut space.reverb_damping_w,
                0.7,
                g,
            );
        }
        Archetype::SkyHush => {
            // `variant` selects the treatment; see the `@q` on the `storm`
            // arm below for the one open question.
            match input.variant {
                1 => {
                    // datarain
                    BusAccum::min_lin(&mut buses[AMB].crush_bits, 10.0, g, 16.0);
                    BusAccum::max_lin(&mut buses[AMB].crush_hold, 2.0, g, 1.0);
                    BusAccum::probabilistic(&mut buses[AMB].crush_mix_survival, 0.2, g);
                    BusAccum::categorical(
                        &mut buses[AMB].hd_mode_score,
                        &mut buses[AMB].hd_mode,
                        score,
                        HoldMode::Stutter,
                    );
                    BusAccum::weighted(
                        &mut buses[AMB].hd_window_wsum,
                        &mut buses[AMB].hd_w,
                        40.0,
                        g,
                    );
                    BusAccum::probabilistic(&mut buses[AMB].hd_mix_survival, 0.12, g);
                }
                2 => {
                    // aurora
                    BusAccum::weighted(&mut buses[MUS].ww_base_wsum, &mut buses[MUS].ww_w, 8.0, g);
                    BusAccum::weighted(&mut buses[MUS].ww_depth_wsum, &mut buses[MUS].ww_w, 2.0, g);
                    BusAccum::categorical(
                        &mut buses[MUS].ww_rate_score,
                        &mut buses[MUS].ww_rate_hz,
                        score,
                        0.1,
                    );
                    BusAccum::probabilistic(&mut buses[MUS].ww_mix_survival, 0.25, g);
                }
                3 => {
                    // watcher: reuses WatcherGaze's recipe, cone-aimed by
                    // the host game's shape.
                    for &b in &[AMB, MUS, SFX, POS] {
                        BusAccum::min_lin(&mut buses[b].width, 0.35, g, 1.0);
                        BusAccum::add_db(&mut buses[b].vol_duck_db, -4.0, g);
                    }
                }
                4 => {
                    // storm: intentionally a no-op.
                    // @q(no sky-flash event exists yet to drive this variant
                    // — add one, or drop `storm` from `variant`'s range?)
                }
                _ => {
                    // void (0, and the fallback)
                    BusAccum::max_lin(&mut buses[AMB].send_db, -60.0, g, OFF_DB);
                    BusAccum::add_db(&mut buses[AMB].hs_db, -6.0, g);
                    BusAccum::add_db(&mut buses[AMB].vol_duck_db, -3.0, g);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> FieldSettings {
        FieldSettings {
            intensity: 1.0,
            reduced: false,
        }
    }
    fn reduced() -> FieldSettings {
        FieldSettings {
            intensity: 1.0,
            reduced: true,
        }
    }

    fn input(archetype: Archetype, weight: f32, depth: f32) -> FieldInput {
        FieldInput {
            archetype,
            weight,
            depth,
            priority: archetype.default_priority(),
            aux: 0.0,
            variant: 0,
            seed: 1,
            bearing: None,
        }
    }

    #[test]
    fn combine_of_empty_is_neutral() {
        assert_eq!(combine(&[], &full()), FieldMix::NEUTRAL);
    }

    #[test]
    fn zero_intensity_is_neutral_even_with_active_fields() {
        let inputs = [input(Archetype::LetheWash, 1.0, 1.0)];
        let mix = combine(
            &inputs,
            &FieldSettings {
                intensity: 0.0,
                reduced: false,
            },
        );
        assert_eq!(mix, FieldMix::NEUTRAL);
    }

    #[test]
    fn archetype_name_roundtrip_covers_all_15() {
        assert_eq!(Archetype::ALL.len(), 15);
        for a in Archetype::ALL {
            assert_eq!(Archetype::from_name(a.name()), Some(a));
        }
        assert_eq!(Archetype::from_name("not_a_real_archetype"), None);
    }

    #[test]
    fn lethe_wash_closes_amb_lowpass_and_leaves_cue_above_its_own_floor() {
        let inputs = [input(Archetype::LetheWash, 1.0, 1.0)];
        let mix = combine(&inputs, &full());
        assert!(mix.buses[AMB].lp_cutoff_hz <= 700.0 + 1.0);
        assert!(mix.buses[AMB].lp_cutoff_hz >= LP_FLOOR_HZ - 1.0);
        assert!(mix.buses[CUE].lp_cutoff_hz >= 2_000.0 - 1.0);
    }

    #[test]
    fn f1_monotonicity_more_weight_closes_amb_lowpass_further() {
        let low = combine(&[input(Archetype::LetheWash, 0.2, 1.0)], &full());
        let high = combine(&[input(Archetype::LetheWash, 0.9, 1.0)], &full());
        assert!(high.buses[AMB].lp_cutoff_hz < low.buses[AMB].lp_cutoff_hz);
    }

    #[test]
    fn f13_monotonicity_more_weight_narrows_width_further() {
        let low = combine(&[input(Archetype::RecognitionNarrow, 0.2, 1.0)], &full());
        let high = combine(&[input(Archetype::RecognitionNarrow, 0.9, 1.0)], &full());
        assert!(high.buses[AMB].width < low.buses[AMB].width);
    }

    #[test]
    fn f8_monotonicity_more_weight_increases_detune_magnitude() {
        let low = combine(&[input(Archetype::ExitDetune, 0.2, 1.0)], &full());
        let high = combine(&[input(Archetype::ExitDetune, 0.9, 1.0)], &full());
        assert!(high.loop_rate_cents.abs() > low.loop_rate_cents.abs());
    }

    #[test]
    fn width_never_drops_below_the_full_mode_floor_under_a_stacked_worst_case() {
        let inputs: Vec<FieldInput> = Archetype::ALL.iter().map(|&a| input(a, 1.0, 1.0)).collect();
        let mix = combine(&inputs, &full());
        for bus in mix.buses {
            assert!(
                bus.width >= WIDTH_FLOOR_FULL - 1e-4,
                "width {} below floor",
                bus.width
            );
        }
    }

    #[test]
    fn wet_mix_sum_never_exceeds_w_max_under_10_000_seeded_random_stacks() {
        // Small deterministic PRNG local to the test, independent of
        // `synth::Lcg` on purpose (a test fixture, not production code).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        for _ in 0..10_000 {
            let n = 1 + (next() * 15.0) as usize;
            let inputs: Vec<FieldInput> = (0..n.min(15))
                .map(|i| {
                    let a = Archetype::ALL[i % 15];
                    let mut inp = input(a, next().clamp(0.0, 1.0), next().clamp(0.0, 1.0));
                    inp.aux = next();
                    inp.variant = if next() > 0.5 { 1 } else { 0 };
                    inp
                })
                .collect();
            let settings = FieldSettings {
                intensity: next().clamp(0.0, 1.0).max(0.01),
                reduced: next() > 0.5,
            };
            let mix = combine(&inputs, &settings);
            let w_max = if settings.reduced {
                W_MAX_REDUCED
            } else {
                W_MAX_FULL
            };
            for bus in mix.buses {
                let sum = bus.crush_mix
                    + bus.ringmod_mix
                    + bus.wow_mix
                    + bus.hold_mix
                    + bus.dist_mix
                    + bus.delay_mix;
                assert!(sum <= w_max + 1e-3, "wet sum {sum} exceeds W_max {w_max}");
                assert!(bus.lp_cutoff_hz >= LP_FLOOR_HZ - 1.0);
                assert!(bus.width >= WIDTH_FLOOR_REDUCED.min(WIDTH_FLOOR_FULL) - 1e-3);
                assert!(bus.vol_duck_db >= VOL_DUCK_FLOOR_DB - 1e-3);
                assert!(bus.crush_dropout <= DROPOUT_CEIL + 1e-3);
            }
            assert!(mix.loop_rate_cents.abs() <= RATE_CEIL_CENTS_FULL + 1e-3);
        }
    }

    #[test]
    fn cue_bus_never_carries_wet_effects_or_dropout() {
        let inputs: Vec<FieldInput> = Archetype::ALL.iter().map(|&a| input(a, 1.0, 1.0)).collect();
        let mix = combine(&inputs, &full());
        let cue = mix.buses[CUE];
        assert_eq!(cue.crush_mix, 0.0);
        assert_eq!(cue.ringmod_mix, 0.0);
        assert_eq!(cue.wow_mix, 0.0);
        assert_eq!(cue.hold_mix, 0.0);
        assert_eq!(cue.dist_mix, 0.0);
        assert_eq!(cue.delay_mix, 0.0);
        assert_eq!(cue.crush_dropout, 0.0);
        assert!(cue.width >= WIDTH_FLOOR_CUE - 1e-4);
        assert!(cue.lp_cutoff_hz >= LP_FLOOR_CUE_HZ - 1.0);
    }

    #[test]
    fn reduced_mode_zeroes_hold_dropout_and_ringmod_and_raises_width_floor() {
        let inputs = [
            input(Archetype::MnemosHold, 1.0, 1.0),
            input(Archetype::LetheWash, 1.0, 1.0),
            input(Archetype::PhantomHollow, 1.0, 1.0),
        ];
        let mix = combine(&inputs, &reduced());
        for bus in mix.buses {
            assert_eq!(bus.hold_mix, 0.0);
            assert_eq!(bus.crush_dropout, 0.0);
            assert_eq!(bus.ringmod_mix, 0.0);
            assert!(bus.width >= WIDTH_FLOOR_REDUCED - 1e-4);
        }
        assert_eq!(mix.space.hold_mix, 0.0);
        assert_eq!(mix.space.hold_mode, HoldMode::Off);
    }

    #[test]
    fn reduced_mode_caps_rate_detune_at_15_cents() {
        let mix = combine(&[input(Archetype::ExitDetune, 1.0, 1.0)], &reduced());
        assert!(mix.loop_rate_cents.abs() <= RATE_CEIL_CENTS_REDUCED + 1e-3);
    }

    #[test]
    fn mnemos_hold_freezes_space_only_when_cycle_held_variant_is_set() {
        let mut stutter = input(Archetype::MnemosHold, 1.0, 1.0);
        stutter.variant = 0;
        let mix_stutter = combine(&[stutter], &full());
        assert_eq!(mix_stutter.space.hold_mode, HoldMode::Stutter);

        let mut held = input(Archetype::MnemosHold, 1.0, 1.0);
        held.variant = 1;
        let mix_held = combine(&[held], &full());
        assert_eq!(mix_held.space.hold_mode, HoldMode::Freeze);
        assert!(mix_held.space.reverb_feedback >= 0.97 - 1e-3);
    }

    #[test]
    fn deterministic_given_the_same_inputs() {
        let inputs = [
            input(Archetype::BitRot, 0.6, 0.7),
            input(Archetype::WatcherGaze, 0.4, 0.5),
        ];
        let a = combine(&inputs, &full());
        let b = combine(&inputs, &full());
        assert_eq!(a, b);
    }

    #[test]
    fn no_field_ever_produces_nan_or_infinite_output_across_the_whole_catalogue() {
        for &a in &Archetype::ALL {
            let mut inp = input(a, 1.0, 1.0);
            inp.aux = 1.0;
            inp.variant = 1;
            let mix = combine(&[inp], &full());
            for bus in mix.buses {
                assert!(bus.ls_db.is_finite());
                assert!(bus.lp_cutoff_hz.is_finite() && bus.lp_cutoff_hz > 0.0);
                assert!(bus.hp_cutoff_hz.is_finite());
                assert!(bus.crush_bits.is_finite());
                assert!(bus.width.is_finite());
                assert!(bus.send_db.is_finite());
            }
            assert!(mix.loop_rate_cents.is_finite());
            assert!(mix.space.reverb_feedback.is_finite());
        }
    }
}
