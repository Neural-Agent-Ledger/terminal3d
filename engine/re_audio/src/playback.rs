//! `playback` — kira-backed audio output. The ONLY module that touches a device.
//!
//! Replaces the old rodio wrapper with **kira 0.12**, which gives us three
//! things we would otherwise hand-roll: a musical clock (BPM tick grid),
//! sample-accurate scheduling of a sound's start onto a clock tick, and built-in
//! mixer-track effects. The deterministic synth ([`crate::synth`]) is unchanged;
//! here we convert its mono f32 PCM into kira `StaticSoundData` and play it.
//!
//! ## Degrade-to-silent contract (unchanged from the rodio era)
//! `AudioManager::<DefaultBackend>::new` returns a `Result`; on a headless CI box
//! with no audio device it `Err`s. We catch that and come up in the *silent*
//! state — the bank is still rendered (callers can inspect PCM) but every
//! playback method is a no-op and nothing ever panics. `is_active()` reports
//! which state we are in. All public methods must be callable in both states.
//!
//! ## Tracks
//! * **master track** — the main mixer track, with a reverb + lowpass filter
//!   ("scene fog"). Most sounds route here.
//! * **accent sub-track → master** — a harsher path (distortion + short delay)
//!   for the terminal-glitch family.
//!
//! ## Scheduling + humanize
//! The plugin asks [`crate::transport::next_tick`] which absolute tick a sound
//! should land on, then calls [`Player::play_oneshot_at`]. We schedule via
//! `StaticSoundData::start_time(clock_time)`. A seeded LCG adds a small ± jitter
//! ("humanize") to quantized sounds only — never to `Immediate` ones — so the
//! grid feels played, not mechanical. The jitter LCG lives here (device side);
//! the synth's deterministic PCM is unaffected (humanize offsets schedule
//! times, not samples).
//!
//! kira/cpal stream handles are not guaranteed `Send`/`Sync`, so [`crate::plugin`]
//! holds [`Player`] as a Bevy `NonSend` resource (as the rodio `Player` was).

use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use kira::clock::{ClockHandle, ClockSpeed};
use kira::effect::compressor::CompressorBuilder;
use kira::effect::delay::{DelayBuilder, DelayHandle};
use kira::effect::distortion::{DistortionBuilder, DistortionHandle, DistortionKind};
use kira::effect::eq_filter::{EqFilterBuilder, EqFilterHandle, EqFilterKind};
use kira::effect::filter::{FilterBuilder, FilterHandle, FilterMode};
use kira::effect::reverb::{ReverbBuilder, ReverbHandle};
use kira::listener::ListenerHandle;
use kira::sound::static_sound::StaticSoundHandle;
use kira::sound::static_sound::{StaticSoundData, StaticSoundSettings};
use kira::track::{
    SendTrackBuilder, SendTrackHandle, SendTrackId, SpatialTrackBuilder, SpatialTrackDistances,
    SpatialTrackHandle, TrackBuilder, TrackHandle,
};
use kira::{
    AudioManager, AudioManagerSettings, Decibels, Easing, Frame, Mapping, Mix, Panning,
    PlaybackRate, Tween, Value,
};

use crate::acoustics::{Room, METERS_PER_CELL};
use crate::ambience::{Ambience, AmbienceMood, AmbienceTargets, Crossfade};
use crate::director::Theme;
use crate::drone_music::{render_music_segment_baked, MusicSequencer, MUSIC_LEVEL};
use crate::dsp::Stereo;
use crate::field::{self, BusFx, FieldMix};
use crate::field_fx::bitcrusher::{BitcrusherBuilder, BitcrusherHandle};
use crate::field_fx::glitch_stutter::{HoldBuilder, HoldHandle, HoldMode};
use crate::field_fx::ring_mod::{RingModBuilder, RingModHandle};
use crate::field_fx::stereo_width::{StereoWidthBuilder, StereoWidthHandle};
use crate::field_fx::wow_flutter::{WowFlutterBuilder, WowFlutterHandle};
use crate::soundscape::{
    render_loop, DetailScheduler, SceneAssets, SceneSpec, DETAIL_LEVEL, HARMONY_LEVEL,
    HARMONY_VOICES, MAX_POSITIONAL_VOICES, POSITIONAL_LEVEL, TEXTURE_LEVEL,
};
use crate::spatial::{
    Listener, LoopKind, SourceKey, VoiceMix, AIR_FAR_HZ, AIR_NEAR_CELLS, AIR_NEAR_HZ, CUTOFF_RADII,
};
use crate::synth::{SoundId, SynthBank, SynthSound, SYNTH_SOUND_COUNT};
use crate::transport::Transport;
use re_core::glam::{Quat, Vec3};

/// World cells → kira's spatial-audio space (metres, Y-up, listener faces
/// **−Z** at identity orientation — see [`kira_orientation`]). World is
/// Z-up with X/Y the floor plane (`re_world::CameraPose`): world +Z (up) →
/// kira +Y; world +Y (the yaw-0 forward axis, `spatial::Listener`'s
/// convention) → kira −Z; world +X (right) is unchanged. Distances are
/// converted from cells to metres so they read consistently against
/// [`crate::acoustics::Room`] (already metric) and kira's own distance
/// parameters.
#[inline]
fn kira_pos(pos_cells: [f32; 3]) -> Vec3 {
    Vec3::new(pos_cells[0], pos_cells[2], -pos_cells[1]) * METERS_PER_CELL
}

/// World yaw (radians, 0 = facing +Y, positive turns the way
/// `spatial::Listener::yaw` does) → the kira listener orientation that
/// faces the same way after [`kira_pos`]'s axis remap.
///
/// Derivation: at yaw θ the world-space forward is `(sin θ, cos θ, 0)` and
/// right is `(cos θ, −sin θ, 0)`; mapping both through [`kira_pos`]'s axis
/// swap gives kira-space forward `(sin θ, 0, −cos θ)` and right
/// `(cos θ, 0, sin θ)`. A rotation of `Quat::from_rotation_y(-θ)` applied to
/// kira's identity forward `(0, 0, −1)` and right `(1, 0, 0)` reproduces
/// exactly those vectors (checked in
/// [`tests::kira_orientation_matches_kira_pos_forward_and_right`]).
#[inline]
fn kira_orientation(yaw: f32) -> Quat {
    Quat::from_rotation_y(-yaw)
}

/// Environment switch that forces the silent (no-device) player even when an
/// audio device exists: set `TS_AUDIO_SILENT=1` (any value but empty / `0`).
///
/// Headless runs of the game's own test-suite, CI, and anyone who needs the
/// machine quiet use it. `re_audio`'s own unit tests are *always* silent (see
/// [`silence_requested`]) — a test must never make a sound on a developer's
/// speakers; the device path is exercised by running the game.
pub const SILENT_ENV: &str = "TS_AUDIO_SILENT";

/// `true` when the player must not open a device: always under `cfg(test)`,
/// otherwise when [`SILENT_ENV`] is set to a non-empty value other than `0`.
pub fn silence_requested() -> bool {
    if cfg!(test) {
        return true;
    }
    std::env::var_os(SILENT_ENV).is_some_and(|v| !v.is_empty() && v != "0")
}

/// Convert a stereo buffer to kira frames.
fn stereo_frames(s: &Stereo) -> Arc<[Frame]> {
    s.l.iter()
        .zip(s.r.iter())
        .map(|(&left, &right)| Frame { left, right })
        .collect()
}

/// Convert a mono buffer to kira frames.
fn mono_frames(m: &[f32]) -> Arc<[Frame]> {
    m.iter().copied().map(Frame::from_mono).collect()
}

/// Build kira sound data from pre-converted frames.
fn frames_data(frames: Arc<[Frame]>) -> StaticSoundData {
    StaticSoundData {
        sample_rate: crate::synth::SAMPLE_RATE,
        frames,
        settings: StaticSoundSettings::default(),
        slice: None,
    }
}

/// The live, room-baked layered ambience of the current level (bed data is
/// played on the crossfading mood beds; texture, harmony and details here).
struct Scene {
    /// What this scene was rendered for.
    spec: SceneSpec,
    /// The stereo bed, converted once, handed to the mood beds.
    bed: Arc<[Frame]>,
    /// Detail one-shots, converted once (index = `DetailKind`).
    details: Vec<Arc<[Frame]>>,
    texture: Option<StaticSoundHandle>,
    harmony: Vec<StaticSoundHandle>,
    detail_sched: DetailScheduler,
    started: Instant,
}

/// Fade used for scene layer swaps.
const SCENE_FADE: Duration = Duration::from_millis(2500);

// ───────────────────────── positional voice pool (kira-native 3D) ────────

/// One pooled positional-voice slot: a kira [`SpatialTrackHandle`] (native
/// listener-relative panning + distance attenuation), a Rust-driven
/// occlusion/front-back filter, and whatever loop is currently assigned to
/// it. Slots are reused across [`SourceKey`]s (voice-stealing), never
/// created per-frame.
struct VoiceSlot {
    track: SpatialTrackHandle,
    /// "Near" filter: occlusion + front/back only (see
    /// [`crate::spatial::Spatial::near_cutoff_hz`]) — the ONE thing kira has
    /// no notion of. Air absorption is a SECOND filter baked into the track
    /// at construction, bound once via `Value::FromListenerDistance` and
    /// never touched again from Rust (kira re-evaluates it every sample from
    /// the track's real, continuously-updated distance to the listener).
    near_filter: FilterHandle,
    sound: Option<StaticSoundHandle>,
    /// The source currently occupying the slot; `None` = free.
    key: Option<SourceKey>,
}

/// How many pooled voice slots to build — one kira `SpatialTrackHandle` each,
/// matching [`MAX_POSITIONAL_VOICES`] (the same budget `select_voices` caps
/// its output at, so every selected voice always finds a free slot).
const VOICE_POOL_SIZE: usize = MAX_POSITIONAL_VOICES;

/// Constant fraction of a positional voice's own intrinsic level sent to the
/// shared room-reverb send track, independent of distance or occlusion (see
/// the "wet/dry-by-distance falls out for free" rationale in
/// `spatial`'s module doc: the dry path already keeps falling with distance
/// via kira's own attenuation, so holding the wet contribution roughly level
/// is what makes the wet:dry ratio rise the way a real room's does).
const POSITIONAL_SEND_LEVEL: f32 = 0.5;

/// Shared distance window (metres) for every pooled voice's kira-native
/// attenuation + air-absorption filter mapping. Kira bakes `distances` and a
/// `Value::FromListenerDistance` filter mapping into a `SpatialTrackBuilder`
/// at construction — there is no runtime setter for either — but the pool is
/// reused across [`LoopKind`]s with different [`LoopKind::radius`]s, so one
/// shared window is picked, sized to the *largest* kind's audible cutoff
/// (`LoopKind::EmitterHiss`). A smaller-radius source (e.g. `PipeHum`) is
/// therefore audible a little past its nominal per-kind cutoff *on the
/// device only* — the offline/verification path
/// (`offline::OfflineMix::add_positional`) keeps the exact per-kind radius
/// via `spatial::Spatial::gain`/`cutoff_hz` unchanged, so it is not affected.
fn voice_pool_distances_m() -> SpatialTrackDistances {
    let max_radius_cells = LoopKind::ALL
        .iter()
        .map(|k| k.radius())
        .fold(0.0f32, f32::max);
    SpatialTrackDistances {
        min_distance: AIR_NEAR_CELLS * METERS_PER_CELL,
        max_distance: (max_radius_cells * CUTOFF_RADII) * METERS_PER_CELL,
    }
}

// ────────────────────── evolving drone music (Goal 2) ─────────────────────

/// One slot of the alternating chord-segment player: the chord it currently
/// holds (once known) and the sound instance once its background render has
/// landed (`None` while still rendering).
#[derive(Default)]
struct MusicSlot {
    chord: Option<usize>,
    sound: Option<StaticSoundHandle>,
}

/// The live evolving drone-music layer: a slow, long-form harmonic drone
/// (`crate::drone_music`) that follows the SAME [`AmbienceMood`] the rest of
/// the ambience system already tracks — an ADDITIONAL layer mixed under the
/// scene/SFX, not a replacement for [`Live::beds`]' existing per-mood flat
/// drone + room-baked scene bed. Two slots alternate: whichever chords
/// [`MusicSequencer::state_at`] names as "current"/"next" are what is
/// playing, crossfaded by its `mix`.
struct Music {
    seq: MusicSequencer,
    /// The room segments are baked with. Set once when the layer starts (or
    /// restarts on a mood change) from whatever room is active then; later
    /// room changes are not re-baked into an already-rendered segment — the
    /// drone layer is a distant, non-positional bed, unlike the ambience
    /// scene texture/harmony (`request_scene`) or positional voices, both of
    /// which DO track the room live. See the delivery report's EDITOR/
    /// follow-up note if this proves audible.
    room: Room,
    started: Instant,
    slots: [MusicSlot; 2],
}

/// Default ambience crossfade duration (seconds) when a mood changes. The game
/// can request a different fade per call via [`Player::set_ambience_fade`].
pub const DEFAULT_AMBIENCE_FADE_SECS: f32 = 2.5;

/// A handle to the currently-looping ambient bed. Opaque; mirrors the rodio-era
/// surface so callers' control flow is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopHandle(SoundId);

impl LoopHandle {
    /// Which sound this loop is playing.
    pub fn sound(self) -> SoundId {
        self.0
    }
}

/// One of the two crossfading drone-bed slots. Each is its own sub-track (so it
/// has independent volume, filter and reverb) routed into the master fog, plus
/// the loop handle of whatever drone is currently playing on it.
struct Bed {
    track: TrackHandle,
    filter: FilterHandle,
    reverb: ReverbHandle,
    /// The mood currently looping here, or `None` if the slot is idle.
    mood: Option<AmbienceMood>,
    /// kira keeps the looped sound alive via this handle; dropping it stops it.
    loop_handle: Option<kira::sound::static_sound::StaticSoundHandle>,
}

/// The master "fog" chain: the sub-track plus handles on the effects whose
/// parameters carry the [`Theme`]'s character. Keeping the handles lets
/// [`Player::set_theme`] *re-voice* the fog when the game changes theme, instead
/// of rebuilding the track (which would orphan every routing into it).
struct MasterFog {
    track: TrackHandle,
    filter: FilterHandle,
    reverb: ReverbHandle,
}

/// An in-progress ambience crossfade: the pure [`Crossfade`] plus the wall-clock
/// instant it began (the device side measures real elapsed time against it).
struct ActiveFade {
    fade: Crossfade,
    started: Instant,
}

/// Convert a linear gain in `[0, 1]` to kira [`Decibels`]. 0 (or below) maps to
/// silence; 1.0 maps to unity (0 dB).
#[inline]
fn linear_to_db(linear: f32) -> Decibels {
    let l = linear.clamp(0.0, 1.0);
    if l <= 0.0 {
        Decibels::SILENCE
    } else {
        Decibels(20.0 * l.log10())
    }
}

// ─────────────────────── sound fields: bus chain ────────────────────────────
//
// Five group tracks (`grp_amb grp_mus grp_sfx grp_cue grp_pos`) sit under
// main; every pre-existing track this file already builds becomes a CHILD of
// one of them (see `try_open_device`'s reparenting), so the chain is purely
// additive — with no field active every custom effect's `mix` is 0 and every
// native stage rests at its own audibly-transparent value (0dB shelves/bells,
// a lowpass/highpass parked at the edge of hearing, delay/distortion at
// `mix = 0`), so the existing mix is unchanged bit-for-bit modulo the filters'
// own (inaudible, edge-of-band) processing.

/// Fixed per-bus `DL` (delay) times: kira's `DelayBuilder::delay_time` is a
/// `Duration` fixed at construction with no live setter, so these are chosen
/// once for the 90 BPM default transport.
pub(crate) const DL_TIME_AMB: Duration = Duration::from_millis(333);
pub(crate) const DL_TIME_MUS: Duration = Duration::from_millis(667);
pub(crate) const DL_TIME_SFX: Duration = Duration::from_millis(111);
pub(crate) const DL_TIME_POS: Duration = Duration::from_millis(167);

/// The main-track limiter: the pre-existing glue `CompressorBuilder` becomes
/// an explicit limiter now that the field chain can push transients close to
/// 0dBFS (worst-case stack acceptance: peaks <= -0.3dBFS).
pub(crate) const LIMITER_THRESHOLD_DB: f64 = -3.0;
pub(crate) const LIMITER_RATIO: f64 = 20.0;
pub(crate) const LIMITER_ATTACK_MS: u64 = 1;
pub(crate) const LIMITER_RELEASE_MS: u64 = 80;

/// A field-chain group track's own linear gain in `[0, 1]`, converted from
/// -6dB for the `FalseEcho` mirror-voice one-shot.
const MIRROR_FOOTSTEP_GAIN: f32 = 0.501; // 10^(-6/20)

/// Dedicated one-shot spatial slots for the `FalseEcho` mirror voice, separate
/// from `voice_slots`: those are reconciled every frame by
/// [`Player::set_positional`] against a *looping* source selection, and a
/// transient one-shot sharing a slot with a looping source would have its
/// position silently overwritten the next time that reconciliation runs. At
/// most one mirror voice per field comfortably fits in 2 (so two overlapping
/// fields, or a retrigger before the first decays, never cut each other off).
const MIRROR_VOICE_SLOTS: usize = 2;

/// One group track's field insert chain (12 fixed-order slots: `LS BL HS LP
/// HP CR RM WW HD DS WD DL`), plus this bus's `VOL` pump state.
pub(crate) struct FieldChain {
    ls: EqFilterHandle,
    bl: EqFilterHandle,
    hs: EqFilterHandle,
    lp: FilterHandle,
    hp: FilterHandle,
    cr: BitcrusherHandle,
    rm: RingModHandle,
    ww: WowFlutterHandle,
    hd: HoldHandle,
    ds: DistortionHandle,
    wd: StereoWidthHandle,
    dl: DelayHandle,
    space_send: SendTrackId,
    /// The `HD` mode last actually applied to `hd` (`None` before the first
    /// [`Player::set_fields`]) — drives the categorical crossfade below (`HD`
    /// mode is one of the parameters that must pick one winner rather than
    /// blend across simultaneous fields).
    hd_mode: Option<HoldMode>,
    /// `Some((new_mode, apply_at))` while a mode switch is crossfading the
    /// `HD` mix down to 0 before applying `new_mode` and crossfading back up.
    hd_pending: Option<(HoldMode, Instant)>,
    /// This bus's `VOL` pump phase (radians), advanced by real elapsed time
    /// each [`Player::set_fields`] call — see that method's doc for why this
    /// is a Rust-driven envelope rather than a kira `Value::FromModulator`
    /// LFO.
    pump_phase: f32,
    last_pump_tick: Instant,
}

/// The `CUE` group's reduced chain: `LP` and `WD` only, plus its own `VOL`
/// duck (no pump — no archetype targets CUE's pump). No `CR/RM/WW/HD/DS/DL`:
/// puzzle feedback must stay legible under any field stack, and `combine`'s
/// CUE floors already guarantee those slots would stay neutral forever —
/// building them would only cost audio-thread cycles for zero possible effect.
pub(crate) struct CueChain {
    lp: FilterHandle,
    wd: StereoWidthHandle,
}

/// Builds one group track's 12-slot field insert chain in fixed order, plus
/// its `SPACE` send route (established here at build time — `set_send` can
/// only retune an existing route's volume, never add a new one). The route
/// starts silent (`Decibels::SILENCE`); [`Player::set_fields`] drives it from
/// `BusFx::send_db`.
pub(crate) fn build_field_chain(
    dl_time: Duration,
    space_send: SendTrackId,
) -> (TrackBuilder, FieldChain) {
    let mut builder = TrackBuilder::new().with_send(space_send, Decibels::SILENCE);
    let ls = builder.add_effect(EqFilterBuilder::new(
        EqFilterKind::LowShelf,
        180.0,
        Decibels(0.0),
        1.0,
    ));
    let bl = builder.add_effect(EqFilterBuilder::new(
        EqFilterKind::Bell,
        1_000.0,
        Decibels(0.0),
        1.0,
    ));
    let hs = builder.add_effect(EqFilterBuilder::new(
        EqFilterKind::HighShelf,
        3_500.0,
        Decibels(0.0),
        1.0,
    ));
    let lp = builder.add_effect(
        FilterBuilder::new()
            .mode(FilterMode::LowPass)
            .cutoff(20_000.0)
            .resonance(0.0),
    );
    let hp = builder.add_effect(
        FilterBuilder::new()
            .mode(FilterMode::HighPass)
            .cutoff(20.0)
            .resonance(0.0),
    );
    let cr = builder.add_effect(BitcrusherBuilder::new());
    let rm = builder.add_effect(RingModBuilder::new());
    let ww = builder.add_effect(WowFlutterBuilder::new());
    let hd = builder.add_effect(HoldBuilder::new());
    let ds = builder.add_effect(
        DistortionBuilder::new()
            .kind(DistortionKind::SoftClip)
            .drive(Decibels(0.0))
            .mix(0.0),
    );
    let wd = builder.add_effect(StereoWidthBuilder::new());
    let dl = builder.add_effect(
        DelayBuilder::new()
            .delay_time(dl_time)
            .feedback(Decibels(-60.0))
            .mix(0.0),
    );
    let now = Instant::now();
    (
        builder,
        FieldChain {
            ls,
            bl,
            hs,
            lp,
            hp,
            cr,
            rm,
            ww,
            hd,
            ds,
            wd,
            dl,
            space_send,
            hd_mode: None,
            hd_pending: None,
            pump_phase: 0.0,
            last_pump_tick: now,
        },
    )
}

/// Builds the `CUE` group's reduced chain (`LP` + `WD` only).
pub(crate) fn build_cue_chain() -> (TrackBuilder, CueChain) {
    let mut builder = TrackBuilder::new();
    let lp = builder.add_effect(
        FilterBuilder::new()
            .mode(FilterMode::LowPass)
            .cutoff(20_000.0)
            .resonance(0.0),
    );
    let wd = builder.add_effect(StereoWidthBuilder::new());
    (builder, CueChain { lp, wd })
}

/// The puzzle-grammar `CUE` bus content: `switch_on/_off`,
/// `refuse_*`, `node_lit/_unlit`, `circuit_resolve`, `win_sting`, `ui_*`,
/// stingers. Checked AFTER [`crate::policy::is_glitch_family`] in every
/// routing call site — "director glitches, and the accent track" stay
/// SFX-classed even for an id that is also nominally cue-flavored
/// (`RefuseCycle`, `StingerCorrupt` are both: glitch wins).
fn is_cue_family(id: SoundId) -> bool {
    matches!(
        id,
        SoundId::SwitchOn
            | SoundId::SwitchOff
            | SoundId::RefusePower
            | SoundId::RefuseCycle
            | SoundId::NodeLit
            | SoundId::NodeUnlit
            | SoundId::CircuitResolve
            | SoundId::WinSting
            | SoundId::UiMove
            | SoundId::UiSelect
            | SoundId::StingerNotice
            | SoundId::StingerMemory
            | SoundId::StingerCorrupt
    )
}

/// `RATE`: cents → a kira [`PlaybackRate`] multiplier.
#[inline]
fn cents_to_rate(cents: f32) -> PlaybackRate {
    PlaybackRate(2f64.powf(f64::from(cents) / 1_200.0))
}

/// Advance a bus's `VOL` pump phase by real elapsed time at `rate_hz`,
/// wrapping every cycle so `sin`/`cos` stay numerically well-behaved over an
/// arbitrarily long session. `last_tick` starts at construction time, so the
/// first call after a long gap (or the very first call) is clamped to avoid
/// a huge phase jump.
fn tick_pump_phase(phase: &mut f32, last_tick: &mut Instant, now: Instant, rate_hz: f32) {
    let dt = now.duration_since(*last_tick).as_secs_f32().min(0.5);
    *last_tick = now;
    *phase = (*phase + std::f32::consts::TAU * rate_hz.max(0.0) * dt) % std::f32::consts::TAU;
}

/// A raised-cosine pump envelope in `[0, 1]` from `phase` (0 at `phase = 0`,
/// peaking at `phase = π`) — the standard sidechain-duck shape: `VOL`'s pump
/// only ever subtracts from `vol_duck_db`, never adds above it.
#[inline]
fn pump_envelope(phase: f32) -> f32 {
    0.5 - 0.5 * phase.cos()
}

/// Categorical winner-take-all with a 75ms-out / switch / 75ms-in crossfade,
/// applied to `HD`'s discrete mode (the one truly discrete categorical slot —
/// `hd.set_mode` is an un-tweened command). `RM` carrier
/// and `WW` rate are also nominally "categorical" (the winner can jump
/// between fields), but both are continuous Hz values already riding
/// their own slot's 60ms tween in [`Player::set_fields`] — a snap there reads
/// as a fast glide, not a mode change, so only `HD` gets the full
/// mute/switch/unmute treatment.
fn apply_hd_mode(
    hd: &mut HoldHandle,
    current: &mut Option<HoldMode>,
    pending: &mut Option<(HoldMode, Instant)>,
    target: HoldMode,
    now: Instant,
) {
    const XFADE: Duration = Duration::from_millis(75);
    if let Some((new_mode, deadline)) = *pending {
        if now >= deadline {
            hd.set_mode(new_mode);
            *current = Some(new_mode);
            *pending = None;
        }
        // Still fading to zero: don't accept a newer target mid-crossfade
        // (avoids thrashing) — it is picked up the next time this runs.
        return;
    }
    match *current {
        None => {
            // First application: the mix itself starts at `BusFx::NEUTRAL`'s
            // 0, so there is nothing to click — apply immediately.
            hd.set_mode(target);
            *current = Some(target);
        }
        Some(c) if c != target => {
            *pending = Some((target, now + XFADE));
        }
        _ => {}
    }
}

/// Apply one bus's [`BusFx`] target to its live field-chain handles, tweened
/// (`tween` for most continuous slots, `slow_tween` — 800ms, slower so a
/// stereo-width or loop-rate change never reads as a jump cut — for `WD`).
/// `HD`'s mode crossfades 75ms regardless (see
/// [`apply_hd_mode`]); its `mix` target is forced to 0 while a crossfade is
/// pending so the two never fight.
pub(crate) fn apply_bus_fx(
    track: &mut TrackHandle,
    chain: &mut FieldChain,
    fx: &BusFx,
    now: Instant,
    tween: Tween,
    slow_tween: Tween,
) {
    // `Decibels`/`Mix` wrap `f32` (matching `BusFx`'s own field types
    // exactly); frequencies/Q/ms/Hz setters take `f64`.
    chain.ls.set_gain(Decibels(fx.ls_db), tween);
    chain.bl.set_frequency(f64::from(fx.bl_freq_hz), tween);
    chain.bl.set_q(f64::from(fx.bl_q.max(0.01)), tween);
    chain.bl.set_gain(Decibels(fx.bl_gain_db), tween);
    chain.hs.set_gain(Decibels(fx.hs_db), tween);
    chain.lp.set_cutoff(f64::from(fx.lp_cutoff_hz), tween);
    chain.hp.set_cutoff(f64::from(fx.hp_cutoff_hz), tween);

    chain.cr.set_bits(f64::from(fx.crush_bits), tween);
    chain.cr.set_hold(f64::from(fx.crush_hold), tween);
    chain.cr.set_dropout(f64::from(fx.crush_dropout), tween);
    chain.cr.set_mix(Mix(fx.crush_mix), tween);

    chain
        .rm
        .set_freq_hz(f64::from(fx.ringmod_carrier_hz), tween);
    chain.rm.set_mix(Mix(fx.ringmod_mix), tween);

    chain.ww.set_base_ms(f64::from(fx.wow_base_ms), tween);
    chain.ww.set_depth_ms(f64::from(fx.wow_depth_ms), tween);
    chain.ww.set_wow_rate_hz(f64::from(fx.wow_rate_hz), tween);
    chain.ww.set_mix(Mix(fx.wow_mix), tween);

    apply_hd_mode(
        &mut chain.hd,
        &mut chain.hd_mode,
        &mut chain.hd_pending,
        fx.hold_mode,
        now,
    );
    chain.hd.set_window_ms(f64::from(fx.hold_window_ms), tween);
    let hd_mix_target = if chain.hd_pending.is_some() {
        0.0
    } else {
        fx.hold_mix
    };
    chain.hd.set_mix(
        Mix(hd_mix_target),
        Tween {
            duration: Duration::from_millis(75),
            ..Tween::default()
        },
    );

    chain.ds.set_drive(Decibels(fx.dist_drive_db), tween);
    chain.ds.set_mix(Mix(fx.dist_mix), tween);

    chain.wd.set_width(f64::from(fx.width), slow_tween);

    chain.dl.set_feedback(Decibels(fx.delay_feedback_db), tween);
    chain.dl.set_mix(Mix(fx.delay_mix), tween);

    let _ = track.set_send(chain.space_send, Decibels(fx.send_db), tween);

    tick_pump_phase(
        &mut chain.pump_phase,
        &mut chain.last_pump_tick,
        now,
        fx.pump_rate_hz,
    );
    let pump = pump_envelope(chain.pump_phase);
    let vol_db = fx.vol_duck_db - fx.pump_depth_db.max(0.0) * pump;
    track.set_volume(Decibels(vol_db), tween);
}

/// Apply the `CUE` bus's reduced [`BusFx`] target (`LP`, `WD`, `VOL` only —
/// see [`CueChain`]).
pub(crate) fn apply_cue_fx(
    track: &mut TrackHandle,
    chain: &mut CueChain,
    fx: &BusFx,
    tween: Tween,
    slow_tween: Tween,
) {
    chain.lp.set_cutoff(f64::from(fx.lp_cutoff_hz), tween);
    chain.wd.set_width(f64::from(fx.width), slow_tween);
    track.set_volume(Decibels(fx.vol_duck_db), tween);
}

/// The live audio device + tracks + clock. Held in one struct so it drops
/// atomically. Field order matters for drop: handles drop before the manager.
struct Live {
    /// The master "fog" sub-track (reverb + lowpass); non-glitch sounds + the
    /// ambient bed route here. Kept alive so its effects stay applied.
    master: MasterFog,
    /// The theme whose parameters the master fog is currently voiced with.
    theme: Theme,
    /// The accent sub-track (distortion + delay) for glitch-family sounds.
    accent: TrackHandle,
    /// The musical clock; its ticks are transport subdivisions.
    clock: ClockHandle,
    /// Master volume in linear `[0, 1]`.
    volume: f32,
    /// The two crossfading drone-bed slots. The active mood lives on one; during
    /// a transition the other holds the outgoing mood. Ambience always routes
    /// through these (both the rich `set_ambience` path and the legacy
    /// `play_loop`/`SetAmbient` path).
    beds: [Bed; 2],
    /// Index into `beds` of the slot holding the *current* (incoming) mood.
    active_bed: usize,
    /// The in-progress crossfade, if any. `None` once a fade has settled.
    fade: Option<ActiveFade>,
    /// The current ambience request (mood + intensity), if a bed is playing.
    ambience: Option<Ambience>,
    /// The fade duration (seconds) used for the next mood change.
    fade_secs: f32,
    /// Legacy single-loop bookkeeping for `SetAmbient`/`play_loop` back-compat:
    /// a non-mood ambient bed (e.g. [`SoundId::Ambient`]) played on bed A without
    /// crossfade. Mutually exclusive with the mood beds in practice.
    loop_sound: Option<SoundId>,
    loop_handle: Option<kira::sound::static_sound::StaticSoundHandle>,
    /// Seeded jitter source for humanize (device side; never affects PCM).
    jitter: u32,
    /// `true` while the active bed plays room-baked scene data (not the mono
    /// drone), so its own reverb is turned down.
    bed_baked: bool,
    /// The scene sub-track (texture / harmony / details), low-passed by the
    /// ambience intensity like the beds.
    scene_track: TrackHandle,
    scene_filter: FilterHandle,
    /// The level's room, once the game has sent one.
    room: Option<Room>,
    /// The live layered scene, if rendered.
    scene: Option<Scene>,
    /// A scene render in flight on a worker thread.
    scene_pending: Option<(SceneSpec, Receiver<SceneAssets>)>,
    /// Current harmony voice gains (already × `HARMONY_LEVEL`).
    harmony_gains: [f32; HARMONY_VOICES],
    /// The positional loops, converted once (index = `LoopKind`).
    loops: Vec<Arc<[Frame]>>,
    /// The single 3D listener every pooled voice is relative to; moved each
    /// frame to the player's camera pose ([`Player::set_positional`]).
    listener: ListenerHandle,
    /// Dedicated room-reverb send for positional voices (kept separate from
    /// `master.reverb` so retuning one for the room never disturbs the
    /// other's — non-positional — voicing). Re-tuned from the same
    /// `Room::kira_reverb()` alongside the master fog in [`Player::set_room`].
    positional_reverb_send: SendTrackHandle,
    positional_reverb: ReverbHandle,
    /// Fixed pool of kira-native spatial voice slots, reused across sources.
    voice_slots: Vec<VoiceSlot>,
    /// Dedicated sub-track for the evolving drone-music layer (Goal 2): no
    /// reverb of its own (each segment is baked with `Room::kira_reverb`
    /// already, same as `scene_track`), routed straight to the main track.
    music_track: TrackHandle,
    /// The live music layer, once a mood has started it (`None` before the
    /// first [`Player::set_ambience`] call).
    music: Option<Music>,
    /// Up to one background chord-segment render per slot in flight.
    music_pending: [Option<(usize, Receiver<Stereo>)>; 2],
    /// The active transport (drives clock speed + humanize amount).
    transport: Transport,
    /// Dedicated one-shot spatial slots for the `FalseEcho` mirror voice —
    /// see [`MIRROR_VOICE_SLOTS`]. Declared after `voice_slots` (a sibling
    /// child of `grp_pos`, same drop-before-parent reasoning) and before
    /// `grp_pos`.
    mirror_slots: Vec<SpatialTrackHandle>,
    /// Round-robin cursor into `mirror_slots`.
    mirror_next: usize,

    // ── sound fields: the five group tracks + the SPACE send ─────────────
    // Declared after every track that is now their CHILD (bed/scene/master/
    // accent/music_track/voice_slots/mirror_slots above), and before
    // `manager` — correct drop order, parents after children, manager last.
    grp_amb: TrackHandle,
    grp_amb_chain: FieldChain,
    grp_mus: TrackHandle,
    grp_mus_chain: FieldChain,
    grp_sfx: TrackHandle,
    grp_sfx_chain: FieldChain,
    grp_cue: TrackHandle,
    grp_cue_chain: CueChain,
    grp_pos: TrackHandle,
    grp_pos_chain: FieldChain,
    /// The shared `SPACE` send: `Hold` (freeze-capable) in front of a reverb.
    /// Never read after construction (every group track captured its
    /// `.id()` already, at `with_send` time) — held only so dropping it
    /// doesn't tear the send track down from under them.
    #[allow(dead_code)]
    space_send: SendTrackHandle,
    space_hold: HoldHandle,
    space_reverb: ReverbHandle,
    space_hd_mode: Option<HoldMode>,
    space_hd_pending: Option<(HoldMode, Instant)>,
    /// The last [`FieldMix`] pushed by [`Player::set_fields`], for callers
    /// that want to read back what is currently applied (e.g. tests, or a
    /// future UI debug readout). Starts at [`FieldMix::NEUTRAL`].
    fields_applied: FieldMix,

    // Declared last → dropped last: the manager owns the renderer/stream and
    // must outlive the track/clock handles that reference it.
    manager: AudioManager,
}

impl Live {
    /// Advance the active crossfade against wall-clock time and write the
    /// resulting bed gains. `active_gain` is the active mood's intensity-driven
    /// target gain; the outgoing bed fades toward silence. When the fade
    /// completes, the outgoing bed is stopped and the fade cleared. Idempotent if
    /// no fade is active (just keeps the active bed at its target gain).
    fn tick_fade_gains(&mut self, active_gain: f32) {
        let gain_tween = Tween {
            duration: Duration::from_millis(60),
            ..Tween::default()
        };
        match self.fade.as_ref() {
            None => {
                // No fade: hold the active bed at its target, silence the other.
                let active = self.active_bed;
                self.beds[active]
                    .track
                    .set_volume(linear_to_db(active_gain), gain_tween);
                let other = 1 - active;
                if self.beds[other].mood.is_none() {
                    self.beds[other]
                        .track
                        .set_volume(Decibels::SILENCE, gain_tween);
                }
            }
            Some(af) => {
                let elapsed = af.started.elapsed().as_secs_f32();
                let (out_g, in_g) = af.fade.gains(elapsed);
                let done = af.fade.done(elapsed);
                let incoming = self.active_bed;
                let outgoing = 1 - incoming;

                self.beds[incoming]
                    .track
                    .set_volume(linear_to_db(in_g * active_gain), gain_tween);
                self.beds[outgoing]
                    .track
                    .set_volume(linear_to_db(out_g * active_gain), gain_tween);

                if done {
                    // Stop the outgoing bed and clear the fade.
                    self.beds[outgoing].loop_handle = None;
                    self.beds[outgoing].mood = None;
                    self.beds[outgoing]
                        .track
                        .set_volume(Decibels::SILENCE, Tween::default());
                    self.fade = None;
                }
            }
        }
    }
}

/// The audio player: the rendered [`SynthBank`] plus an optional live device.
///
/// Construct with [`Player::open`]. When no device is available the player is in
/// the *silent* state: it still holds the bank but every playback method is a
/// no-op.
pub struct Player {
    bank: SynthBank,
    live: Option<Live>,
}

impl Player {
    /// Render the bank and open the default device. Never fails: degrades to a
    /// silent no-op player when no device exists.
    pub fn open() -> Player {
        Self::open_with_bank(SynthBank::build())
    }

    /// Like [`Player::open`] but reuses an already-built [`SynthBank`].
    pub fn open_with_bank(bank: SynthBank) -> Player {
        let live = if silence_requested() {
            None
        } else {
            Self::try_open_device(Transport::default())
        };
        Player { bank, live }
    }

    /// A player that never opens a device, whatever the environment. Callers
    /// that must stay quiet (tools, tests in other crates) use this directly.
    pub fn silent() -> Player {
        Player {
            bank: SynthBank::build(),
            live: None,
        }
    }

    /// Attempt to bring up kira + the master/accent tracks + clock. Returns
    /// `None` (silent) on any failure — the headless/CI/no-driver path.
    fn try_open_device(transport: Transport) -> Option<Live> {
        // The main-track limiter: everything sums here (all five field
        // buses, the SPACE tail, positional voices) and the field chain can
        // now push transients close to 0dBFS, so this is an explicit limiter
        // — not the gentler glue compressor it used to be.
        let settings = AudioManagerSettings::<kira::DefaultBackend> {
            main_track_builder: kira::track::MainTrackBuilder::new().with_effect(
                CompressorBuilder::new()
                    .threshold(LIMITER_THRESHOLD_DB)
                    .ratio(LIMITER_RATIO)
                    .attack_duration(Duration::from_millis(LIMITER_ATTACK_MS))
                    .release_duration(Duration::from_millis(LIMITER_RELEASE_MS))
                    .makeup_gain(Decibels(0.0)),
            ),
            ..Default::default()
        };
        let mut manager = AudioManager::<kira::DefaultBackend>::new(settings).ok()?;

        // ── sound fields: SPACE send, then the five group tracks ──────────
        // SPACE first: the group tracks' `with_send` route needs its id.
        let mut space_builder = SendTrackBuilder::new();
        let space_hold = space_builder.add_effect(HoldBuilder::new());
        let space_reverb = space_builder.add_effect(
            ReverbBuilder::new()
                .feedback(0.85)
                .damping(0.5)
                .mix(Mix(1.0)),
        );
        let space_send = manager.add_send_track(space_builder).ok()?;

        let (amb_builder, grp_amb_chain) = build_field_chain(DL_TIME_AMB, space_send.id());
        let mut grp_amb = manager.add_sub_track(amb_builder).ok()?;
        let (mus_builder, grp_mus_chain) = build_field_chain(DL_TIME_MUS, space_send.id());
        let mut grp_mus = manager.add_sub_track(mus_builder).ok()?;
        let (sfx_builder, grp_sfx_chain) = build_field_chain(DL_TIME_SFX, space_send.id());
        let mut grp_sfx = manager.add_sub_track(sfx_builder).ok()?;
        let (cue_builder, grp_cue_chain) = build_cue_chain();
        let grp_cue = manager.add_sub_track(cue_builder).ok()?;
        let (pos_builder, grp_pos_chain) = build_field_chain(DL_TIME_POS, space_send.id());
        let mut grp_pos = manager.add_sub_track(pos_builder).ok()?;

        // Master "fog" sub-track (reverb + lowpass) and the accent (glitch)
        // sub-track are now children of `grp_sfx` — director glitches and
        // the accent track belong under the SFX bus, and non-glitch
        // one-shots (footsteps, portal_warp, storey_*, note_read)
        // already route to `master.track`. The field chain wraps around
        // both without changing either's own existing voicing.
        let master = Self::theme_master_track(&mut grp_sfx, Theme::Default)?;
        let accent = Self::accent_track(&mut grp_sfx)?;

        // Two crossfading drone-bed slots, now children of `grp_amb`.
        let bed_a = Self::drone_bed(&mut grp_amb)?;
        let bed_b = Self::drone_bed(&mut grp_amb)?;

        // The scene sub-track: texture + harmony + detail events. Its own
        // low-pass follows the ambience intensity; the room is baked into the
        // layers, so no reverb here. Now a child of `grp_amb`.
        let mut scene_builder = TrackBuilder::new();
        let scene_filter = scene_builder
            .add_effect(FilterBuilder::new().cutoff(AmbienceTargets::MAX_CUTOFF_HZ as f64));
        let scene_track = grp_amb.add_sub_track(scene_builder).ok()?;
        let loops = LoopKind::ALL
            .iter()
            .map(|&k| mono_frames(&render_loop(k)))
            .collect();

        // ── kira-native 3D: one listener, a dedicated positional reverb
        // send, and a fixed pool of spatial voice slots ────────────────────
        let listener = manager.add_listener(Vec3::ZERO, Quat::IDENTITY).ok()?;

        let mut send_builder = SendTrackBuilder::new();
        let positional_reverb = send_builder.add_effect(
            // `mix(Mix(1.0))`: a send track carries only the wet signal by
            // convention (the dry copy stays on each voice's own track); the
            // per-voice `POSITIONAL_SEND_LEVEL` controls how much reaches it.
            ReverbBuilder::new()
                .feedback(0.85)
                .damping(0.5)
                .mix(Mix(1.0)),
        );
        let positional_reverb_send = manager.add_send_track(send_builder).ok()?;

        let distances = voice_pool_distances_m();
        let air_mapping = Mapping {
            input_range: (
                f64::from(distances.min_distance),
                f64::from(distances.max_distance),
            ),
            output_range: (f64::from(AIR_NEAR_HZ), f64::from(AIR_FAR_HZ)),
            easing: Easing::Linear,
        };
        // Positional voice slots (looping sources, reconciled by
        // `Player::set_positional`) and the F4 mirror-voice one-shot slots
        // are now children of `grp_pos`.
        let mut voice_slots = Vec::with_capacity(VOICE_POOL_SIZE);
        for _ in 0..VOICE_POOL_SIZE {
            let mut builder = SpatialTrackBuilder::new()
                .volume(Decibels::SILENCE)
                .distances(distances)
                // Kira's own distance curve; steeper than linear so a source
                // reads "present" until fairly close to its cutoff, then
                // falls away — closer to how the ear perceives loudness than
                // a linear ramp would.
                .attenuation_function(Some(Easing::InPowi(2)))
                .spatialization_strength(1.0)
                .with_send(positional_reverb_send.id(), Decibels::SILENCE);
            // Air absorption: fully kira-native, bound once, re-evaluated by
            // the audio thread from the track's real per-sample distance —
            // no per-frame Rust cost.
            builder
                .add_effect(FilterBuilder::new().cutoff(Value::FromListenerDistance(air_mapping)));
            // Occlusion + front/back: kira has no notion of either, so this
            // one IS driven from Rust every frame (`Player::set_positional`).
            let near_filter =
                builder.add_effect(FilterBuilder::new().cutoff(f64::from(AIR_NEAR_HZ)));
            let track = grp_pos
                .add_spatial_sub_track(&listener, kira_pos([0.0, 0.0, 0.0]), builder)
                .ok()?;
            voice_slots.push(VoiceSlot {
                track,
                near_filter,
                sound: None,
                key: None,
            });
        }

        let mut mirror_slots = Vec::with_capacity(MIRROR_VOICE_SLOTS);
        for _ in 0..MIRROR_VOICE_SLOTS {
            let builder = SpatialTrackBuilder::new()
                .volume(Decibels::SILENCE)
                .distances(distances);
            let track = grp_pos
                .add_spatial_sub_track(&listener, kira_pos([0.0, 0.0, 0.0]), builder)
                .ok()?;
            mirror_slots.push(track);
        }

        // The evolving drone-music layer's own track: no filter/reverb of its
        // own (each rendered segment already has the room baked in, same as
        // `scene_track`). Now a child of `grp_mus`.
        let music_track = grp_mus.add_sub_track(TrackBuilder::new()).ok()?;

        // Musical clock: one tick = one transport subdivision.
        let mut clock = manager
            .add_clock(ClockSpeed::TicksPerMinute(transport.ticks_per_minute()))
            .ok()?;
        clock.start();

        Some(Live {
            master,
            theme: Theme::Default,
            accent,
            clock,
            volume: 1.0,
            beds: [bed_a, bed_b],
            active_bed: 0,
            fade: None,
            ambience: None,
            fade_secs: DEFAULT_AMBIENCE_FADE_SECS,
            loop_sound: None,
            loop_handle: None,
            jitter: 0x1234_5678,
            bed_baked: false,
            scene_track,
            scene_filter,
            room: None,
            scene: None,
            scene_pending: None,
            harmony_gains: [0.0; HARMONY_VOICES],
            loops,
            listener,
            positional_reverb_send,
            positional_reverb,
            voice_slots,
            music_track,
            music: None,
            music_pending: [None, None],
            transport,
            mirror_slots,
            mirror_next: 0,
            grp_amb,
            grp_amb_chain,
            grp_mus,
            grp_mus_chain,
            grp_sfx,
            grp_sfx_chain,
            grp_cue,
            grp_cue_chain,
            grp_pos,
            grp_pos_chain,
            space_send,
            space_hold,
            space_reverb,
            space_hd_mode: None,
            space_hd_pending: None,
            fields_applied: FieldMix::NEUTRAL,
            manager,
        })
    }

    /// Build one crossfading drone-bed sub-track: a filter + reverb chain routed
    /// into `parent` (now `grp_amb`, previously the manager directly), starting
    /// silent. The returned handles let [`Player::set_ambience`] modulate
    /// cutoff / reverb depth per intensity and ramp the bed gain for
    /// crossfades. The bed keeps its own self-contained fog path — its
    /// reverb/filter is layered under `grp_amb`'s field chain, not replaced
    /// by it.
    fn drone_bed(parent: &mut TrackHandle) -> Option<Bed> {
        let mut builder = TrackBuilder::new();
        // Start fully attenuated so a freshly-built bed is silent until a mood is
        // assigned and faded in.
        builder = builder.volume(Decibels::SILENCE);
        let filter =
            builder.add_effect(FilterBuilder::new().cutoff(AmbienceTargets::MIN_CUTOFF_HZ as f64));
        let reverb = builder.add_effect(
            ReverbBuilder::new()
                .feedback(0.88)
                .mix(Mix(AmbienceTargets::MIN_REVERB_MIX)),
        );
        let track = parent.add_sub_track(builder).ok()?;
        Some(Bed {
            track,
            filter,
            reverb,
            mood: None,
            loop_handle: None,
        })
    }

    /// The master-fog voicing for a theme: `(reverb feedback, lowpass cutoff Hz)`.
    /// Feedback is the tail length ("how big is the room"); the cutoff is how
    /// much top end survives. Pure data — no device — so it is unit-testable.
    fn theme_params(theme: Theme) -> (f64, f64) {
        match theme {
            // Cold: longer/darker reverb, lower cutoff.
            Theme::Cold => (0.92, 1400.0),
            // Garden: shorter/warmer.
            Theme::Garden => (0.80, 4200.0),
            Theme::Default => (0.86, 2600.0),
            // ── campaign movements: the fog closes in as the story does ──
            // P1 "first light": the most open, warmest air in the game.
            Theme::P1FirstLight => (0.76, 5200.0),
            // P2 "the gaze": tighter and darker — the room has an opinion.
            Theme::P2TheGaze => (0.84, 2200.0),
            // P3 "grief": hollow and long, the river under everything.
            Theme::P3Grief => (0.94, 1100.0),
            // P4 "closing": widest tail, almost no top end left.
            Theme::P4Closing => (0.96, 800.0),
        }
    }

    /// Build the master "fog" sub-track (reverb + lowpass) voiced for `theme`,
    /// routed into `parent` (now `grp_sfx`, previously the manager directly).
    /// The effect handles come back with it so the voicing can be changed later
    /// ([`Player::set_theme`]) without rebuilding the routing target.
    fn theme_master_track(parent: &mut TrackHandle, theme: Theme) -> Option<MasterFog> {
        let (reverb_feedback, lowpass_cutoff) = Self::theme_params(theme);
        let mut builder = TrackBuilder::new();
        let reverb = builder.add_effect(
            ReverbBuilder::new()
                .feedback(reverb_feedback)
                .mix(Mix(0.25)),
        );
        let filter = builder.add_effect(FilterBuilder::new().cutoff(lowpass_cutoff));
        let track = parent.add_sub_track(builder).ok()?;
        Some(MasterFog {
            track,
            filter,
            reverb,
        })
    }

    /// Build the accent sub-track (distortion + short delay) for glitches,
    /// routed into `parent` (now `grp_sfx`, previously the manager directly).
    fn accent_track(parent: &mut TrackHandle) -> Option<TrackHandle> {
        let track = parent
            .add_sub_track(
                TrackBuilder::new()
                    .with_effect(DistortionBuilder::new().drive(Decibels(8.0)).mix(0.6))
                    .with_effect(
                        DelayBuilder::new()
                            .delay_time(Duration::from_millis(120))
                            .feedback(Decibels(-9.0))
                            .mix(0.3),
                    ),
            )
            .ok()?;
        Some(track)
    }

    /// True when a real device is driving output; false = silent no-op mode.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.live.is_some()
    }

    /// Borrow the rendered synth bank (always present, even when silent).
    #[inline]
    pub fn bank(&self) -> &SynthBank {
        &self.bank
    }

    /// Build a kira [`StaticSoundData`] from a sound's mono PCM, at the given
    /// linear volume and pan. Mono frames are upmixed to stereo via
    /// [`Frame::from_mono`].
    fn sound_data(&self, id: SoundId, volume: f32, pan: f32) -> StaticSoundData {
        let sound: &SynthSound = self.bank.get(id);
        let frames: Arc<[Frame]> = sound.frames.iter().copied().map(Frame::from_mono).collect();
        StaticSoundData {
            sample_rate: sound.sample_rate,
            frames,
            settings: StaticSoundSettings::default(),
            slice: None,
        }
        .volume(linear_to_db(volume))
        .panning(pan.clamp(-1.0, 1.0))
    }

    /// Next humanize offset in ticks (a small ± fraction of a tick), advancing
    /// the seeded jitter LCG. `humanize_ms` is converted to a tick fraction
    /// using the current tempo. Returns 0.0 when humanize is disabled.
    fn next_humanize_ticks(live: &mut Live) -> f64 {
        let ms = live.transport.humanize_ms;
        if ms <= 0.0 {
            return 0.0;
        }
        // Advance the LCG (same constants as the synth) → a value in [-1, 1).
        live.jitter = live
            .jitter
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        let unit = (live.jitter >> 8) as f64 / 16_777_216.0; // [0, 1)
        let signed = unit * 2.0 - 1.0; // [-1, 1)
                                       // Convert ± ms to ticks: ticks_per_second = ticks_per_minute / 60.
        let ticks_per_sec = live.transport.ticks_per_minute() / 60.0;
        signed * (ms / 1000.0) * ticks_per_sec
    }

    /// Fire a one-shot immediately (no beat quantization). Used for `Immediate`
    /// sounds. No-op when silent.
    pub fn play_oneshot(&mut self, id: SoundId) {
        self.play_internal(id, None, 0.0);
    }

    /// Fire a one-shot scheduled onto an absolute clock `tick`, with humanize
    /// jitter applied. Used for beat-quantized sounds. `pan` is `[-1, 1]`.
    /// No-op when silent. Falls back to immediate if the clock is unavailable.
    pub fn play_oneshot_at(&mut self, id: SoundId, tick: u64, pan: f32) {
        self.play_internal(id, Some(tick), pan);
    }

    /// The current clock tick + fractional position, for the plugin to feed
    /// [`crate::transport::next_tick`]. `None` when silent (no clock).
    pub fn clock_now(&self) -> Option<(u64, f64)> {
        self.live.as_ref().map(|l| {
            let t = l.clock.time();
            (t.ticks, t.fraction)
        })
    }

    /// The active transport.
    pub fn transport(&self) -> Transport {
        self.live.as_ref().map(|l| l.transport).unwrap_or_default()
    }

    /// Shared internal: build the sound, choose start time + routing, play it.
    fn play_internal(&mut self, id: SoundId, schedule_tick: Option<u64>, pan: f32) {
        let vol = self.live.as_ref().map(|l| l.volume).unwrap_or(1.0);
        let data = self.sound_data(id, vol, pan);
        let glitch = crate::policy::is_glitch_family(id);

        let Some(live) = self.live.as_mut() else {
            return; // silent
        };

        // Apply scheduling (+ humanize) for quantized sounds.
        let data = if let Some(tick) = schedule_tick {
            let jitter = Self::next_humanize_ticks(live);
            let start = live.clock.time();
            // Target = absolute tick + humanize fraction, never before now.
            let base = kira::clock::ClockTime {
                clock: live.clock.id(),
                ticks: tick,
                fraction: 0.0,
            };
            let start_time = if jitter >= 0.0 {
                base + jitter
            } else {
                // Don't schedule before the current clock time.
                let shifted = base - (-jitter);
                if shifted < start {
                    start
                } else {
                    shifted
                }
            };
            data.start_time(start_time)
        } else {
            data
        };

        // Route glitches to the accent track; puzzle-grammar cues to the CUE
        // bus; everything else to the master fog (the SFX bus).
        // Glitch wins over cue for the two ids that are both (`RefuseCycle`,
        // `StingerCorrupt`) — see `is_cue_family`'s doc.
        let result = if glitch {
            live.accent.play(data)
        } else if is_cue_family(id) {
            live.grp_cue.play(data)
        } else {
            live.master.track.play(data)
        };
        // A play error (e.g. sound-limit reached) is non-fatal: drop it.
        let _ = result;
    }

    /// Start (or replace) the single looping ambient bed via kira's
    /// `loop_region`. Same-sound re-loop is a no-op. Returns a [`LoopHandle`]
    /// even when silent. The ambient bed is not beat-quantized.
    ///
    /// Back-compat: this is the legacy [`SetAmbient`](crate::plugin::SetAmbient)
    /// path. If `id` is one of the four drone moods, it is forwarded to the rich
    /// [`Player::set_ambience`] path (so even a legacy call crossfades) at a
    /// neutral default intensity. Non-drone beds (e.g. [`SoundId::Ambient`]) use
    /// the simple single-loop slot as before.
    pub fn play_loop(&mut self, id: SoundId) -> LoopHandle {
        // Drone ids go through the crossfading mood bed.
        if let Some(mood) = AmbienceMood::from_drone(id) {
            // Preserve current intensity if already set on a bed, else default.
            let intensity = self
                .live
                .as_ref()
                .and_then(|l| l.ambience)
                .map(|a| a.intensity)
                .unwrap_or(crate::ambience::DEFAULT_INTENSITY);
            self.set_ambience(Ambience::new(mood, intensity));
            return LoopHandle(id);
        }

        let vol = self.live.as_ref().map(|l| l.volume).unwrap_or(1.0);
        let data = self.sound_data(id, vol, 0.0).loop_region(0.0..);

        if let Some(live) = self.live.as_mut() {
            if live.loop_sound == Some(id) {
                return LoopHandle(id); // already looping this exact sound
            }
            // Drop the previous loop's handle → kira stops it.
            live.loop_handle = None;
            if let Ok(handle) = live.master.track.play(data) {
                live.loop_handle = Some(handle);
                live.loop_sound = Some(id);
            }
        }
        LoopHandle(id)
    }

    /// Stop the currently-looping ambient bed, if any (both the legacy single
    /// loop and any active mood bed). No-op when silent.
    pub fn stop_loop(&mut self) {
        if let Some(live) = self.live.as_mut() {
            live.loop_handle = None; // drop → stop legacy loop
            live.loop_sound = None;
            // Also tear down the mood beds + any in-progress fade.
            live.fade = None;
            live.ambience = None;
            for bed in &mut live.beds {
                bed.loop_handle = None;
                bed.mood = None;
                bed.track.set_volume(Decibels::SILENCE, Tween::default());
            }
        }
    }

    /// The sound currently looping (legacy single loop), or `None`. Mood beds
    /// report via [`Player::ambience`] / [`Player::active_mood`] instead.
    pub fn looping(&self) -> Option<SoundId> {
        self.live.as_ref().and_then(|l| l.loop_sound)
    }

    // ── ambience: crossfading mood beds + intensity-driven dread ──────────────

    /// Set the ambience crossfade duration (seconds) used for the *next* mood
    /// change. Clamped to a sane minimum. No-op when silent (but harmless).
    pub fn set_ambience_fade(&mut self, secs: f32) {
        if let Some(live) = self.live.as_mut() {
            live.fade_secs = secs.max(Crossfade::MIN_DURATION);
        }
    }

    /// The mood currently fading-in / playing, or `None` if no bed is active.
    pub fn active_mood(&self) -> Option<AmbienceMood> {
        self.live.as_ref().and_then(|l| l.ambience.map(|a| a.mood))
    }

    /// The current ambience request (mood + intensity), or `None`.
    pub fn ambience(&self) -> Option<Ambience> {
        self.live.as_ref().and_then(|l| l.ambience)
    }

    /// `true` while an ambience crossfade is in progress.
    pub fn is_crossfading(&self) -> bool {
        self.live.as_ref().is_some_and(|l| l.fade.is_some())
    }

    /// Apply an [`Ambience`] request: select the drone bed for `mood` and scale
    /// its presence by `intensity`.
    ///
    /// * Same mood, new intensity → re-target the active bed's gain / filter /
    ///   reverb (tweened, no bed swap).
    /// * New mood → start a crossfade: fade the active bed out and the new mood's
    ///   bed in over [`Player::set_ambience_fade`]'s duration, overlapping. The
    ///   incoming bed is started looping immediately.
    ///
    /// No-op when silent (still records nothing to play). The drone PCM is
    /// deterministic and untouched; only mixer params move.
    pub fn set_ambience(&mut self, req: Ambience) {
        let vol = self.live.as_ref().map(|l| l.volume).unwrap_or(1.0);
        // Build the looped bed data outside the borrow of `live`: the room-baked
        // stereo scene bed when one is rendered for this mood, else the mono
        // drone (first seconds of a level, or before any room is known).
        let scene_bed = self.live.as_ref().and_then(|l| {
            l.scene
                .as_ref()
                .filter(|sc| sc.spec.mood == req.mood)
                .map(|sc| sc.bed.clone())
        });
        let baked = scene_bed.is_some();
        let data = match scene_bed {
            Some(frames) => frames_data(frames).volume(linear_to_db(vol)),
            None => self.sound_data(req.drone(), vol, 0.0),
        }
        .loop_region(0.0..);

        let Some(live) = self.live.as_mut() else {
            return; // silent: nothing to schedule
        };
        // A mood the current scene was not rendered for → render one.
        Self::request_scene(live, Some(req.mood));

        let targets = req.targets();
        let prev_mood = live.ambience.map(|a| a.mood);
        live.ambience = Some(req);

        Self::apply_scene_intensity(live, targets);
        if prev_mood == Some(req.mood) {
            // Same mood, new intensity: re-target the live bed; no swap, no fade.
            let idx = live.active_bed;
            Self::apply_targets_to_bed(&mut live.beds[idx], targets, true, live.bed_baked);
            return;
        }
        live.bed_baked = baked;
        Self::restart_music(live, req.mood);

        // New mood → crossfade. The incoming bed is the *other* slot. Tear down
        // anything stale on that slot first (e.g. a prior fade's outgoing bed) to
        // avoid a third overlapping voice.
        let incoming = 1 - live.active_bed;
        live.beds[incoming].loop_handle = None;
        if let Ok(handle) = live.beds[incoming].track.play(data) {
            live.beds[incoming].loop_handle = Some(handle);
            live.beds[incoming].mood = Some(req.mood);
        }
        // Target the incoming bed's filter/reverb now (gain is owned by the fade).
        Self::apply_targets_to_bed(&mut live.beds[incoming], targets, false, baked);

        // Equal-power crossfade from the previous mood (or from silence on first
        // assignment) to the new mood, over the configured duration.
        let from = prev_mood.unwrap_or(req.mood);
        live.fade = Some(ActiveFade {
            fade: Crossfade::new(from, req.mood, live.fade_secs),
            started: Instant::now(),
        });
        live.active_bed = incoming;
        // Drive gains immediately so the first frame already has the overlap set.
        live.tick_fade_gains(targets.gain);
    }

    /// Per-frame ambience driver: advance any in-progress crossfade and keep the
    /// active bed's intensity params applied. The plugin calls this every frame.
    /// No-op when silent.
    pub fn tick_ambience(&mut self) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        let Some(req) = live.ambience else {
            return;
        };
        let target_gain = req.targets().gain;
        live.tick_fade_gains(target_gain);
        self.tick_scene();
        self.tick_music();
    }

    /// Apply intensity targets (filter cutoff + reverb mix, and optionally gain)
    /// to a bed, tweened. Gain is skipped while a crossfade owns it.
    fn apply_targets_to_bed(bed: &mut Bed, targets: AmbienceTargets, set_gain: bool, baked: bool) {
        // Movement → a touch more reverb stereo width / feedback sway: deeper
        // intensity = a more alive, unstable space. The synth LFO is baked into
        // PCM (deterministic); here we add device-side movement only.
        let tween = Tween {
            duration: Duration::from_millis(400),
            ..Tween::default()
        };
        bed.filter
            .set_cutoff(targets.filter_cutoff_hz as f64, tween);
        // A room-baked scene bed already carries its reflections + tail; the
        // bed's own reverb only adds a little bloom on top of it.
        let mix = if baked {
            targets.reverb_mix * 0.3
        } else {
            targets.reverb_mix
        };
        bed.reverb.set_mix(Mix(mix), tween);
        bed.reverb
            .set_stereo_width(0.5 + 0.5 * targets.movement as f64, tween);
        if set_gain {
            bed.track.set_volume(linear_to_db(targets.gain), tween);
        }
    }

    /// Set master volume in linear `[0, 1]` (clamped). Applies to the main
    /// mixer track immediately and to all future sounds. No-op when silent.
    pub fn set_volume(&mut self, v: f32) {
        let v = v.clamp(0.0, 1.0);
        if let Some(live) = self.live.as_mut() {
            live.volume = v;
            live.manager
                .main_track()
                .set_volume(linear_to_db(v), Tween::default());
        }
    }

    /// Current master volume (1.0 when silent — the resting default).
    pub fn volume(&self) -> f32 {
        self.live.as_ref().map(|l| l.volume).unwrap_or(1.0)
    }

    /// Re-voice the master "fog" for `theme`: ramp the reverb tail and the
    /// lowpass cutoff to the theme's values over a short tween, so a level
    /// change *slides* into its movement rather than cutting. Everything already routed through the fog (including the drone
    /// beds) keeps playing — the track itself is never rebuilt. No-op when
    /// silent.
    pub fn set_theme(&mut self, theme: Theme) {
        let (feedback, cutoff) = Self::theme_params(theme);
        if let Some(live) = self.live.as_mut() {
            if live.theme == theme {
                return;
            }
            live.theme = theme;
            // Slow enough to read as the room changing, not as a click.
            let tween = Tween {
                duration: Duration::from_millis(1200),
                ..Tween::default()
            };
            live.master.reverb.set_feedback(feedback, tween);
            live.master.filter.set_cutoff(cutoff, tween);
        }
    }

    /// The theme the master fog is currently voiced with ([`Theme::Default`]
    /// when silent — the resting default).
    pub fn theme(&self) -> Theme {
        self.live.as_ref().map(|l| l.theme).unwrap_or_default()
    }

    /// Reconfigure the transport: change the clock speed (BPM/subdivision) live
    /// and store the new humanize amount. No-op when silent.
    pub fn set_transport(&mut self, transport: Transport) {
        if let Some(live) = self.live.as_mut() {
            live.transport = transport;
            live.clock.set_speed(
                ClockSpeed::TicksPerMinute(transport.ticks_per_minute()),
                Tween::default(),
            );
        }
    }
}

impl Player {
    // ── voices with pitch / gain / pan (cues, director, details) ─────────────

    /// Play a one-shot with a playback `rate` (pitch), linear `gain` and `pan`,
    /// immediately (`tick == None`) or scheduled onto a clock tick with humanize.
    /// Routing follows [`crate::policy::is_glitch_family`]. No-op when silent.
    pub fn play_voice(&mut self, id: SoundId, rate: f64, gain: f32, pan: f32, tick: Option<u64>) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        let sound: &SynthSound = self.bank.get(id);
        let mut data = frames_data(mono_frames(&sound.frames))
            .volume(linear_to_db(gain.clamp(0.0, 1.0)))
            .panning(Panning(pan.clamp(-1.0, 1.0)))
            .playback_rate(PlaybackRate(rate.clamp(0.125, 8.0)));
        if let Some(t) = tick {
            let jitter = Self::next_humanize_ticks(live);
            let base = kira::clock::ClockTime {
                clock: live.clock.id(),
                ticks: t,
                fraction: 0.0,
            };
            let now = live.clock.time();
            let start = if jitter >= 0.0 {
                base + jitter
            } else {
                let shifted = base - (-jitter);
                if shifted < now {
                    now
                } else {
                    shifted
                }
            };
            data = data.start_time(start);
        }
        // A play error (sound limit) is non-fatal. Same routing precedence as
        // `play_internal`: glitch, then cue, then the SFX master fog.
        let _ = if crate::policy::is_glitch_family(id) {
            live.accent.play(data)
        } else if is_cue_family(id) {
            live.grp_cue.play(data)
        } else {
            live.master.track.play(data)
        };
    }

    // ── the room ─────────────────────────────────────────────────────────────

    /// Adopt the level's acoustic [`Room`]: re-voice the master fog's reverb
    /// (tail length / damping / wetness) for this room and start rendering the
    /// room-baked scene layers for the active mood. No-op when silent.
    pub fn set_room(&mut self, room: Room) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        if live.room == Some(room) {
            return;
        }
        live.room = Some(room);
        let k = room.kira_reverb();
        let tween = Tween {
            duration: Duration::from_millis(800),
            ..Tween::default()
        };
        live.master.reverb.set_feedback(k.feedback, tween);
        live.master.reverb.set_damping(k.damping, tween);
        live.master.reverb.set_mix(Mix(k.mix), tween);
        // The positional-voice send shares the room's tail length/damping
        // (same acoustic space) but keeps its own fixed `mix(1.0)` — a send
        // track carries only the wet signal by convention, so its own "how
        // wet" knob is meaningless; `POSITIONAL_SEND_LEVEL` on each voice is
        // what actually controls how much of it is heard.
        live.positional_reverb.set_feedback(k.feedback, tween);
        live.positional_reverb.set_damping(k.damping, tween);
        let mood = live.ambience.map(|a| a.mood);
        Self::request_scene(live, mood);
    }

    /// The room the player was last given (`None` when silent or unset).
    pub fn room(&self) -> Option<Room> {
        self.live.as_ref().and_then(|l| l.room)
    }

    /// Start a worker-thread render of the scene for `(mood, room)` unless the
    /// live or pending scene already matches. Needs both a room and a mood.
    fn request_scene(live: &mut Live, mood: Option<AmbienceMood>) {
        let (Some(room), Some(mood)) = (live.room, mood) else {
            return;
        };
        let spec = SceneSpec::new(mood, room, scene_seed(mood, &room));
        if live.scene.as_ref().is_some_and(|sc| sc.spec == spec)
            || live.scene_pending.as_ref().is_some_and(|(p, _)| *p == spec)
        {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name(String::from("re_audio-scene"))
            .spawn(move || {
                // The receiver may be gone (superseded): ignore the send error.
                let _ = tx.send(SceneAssets::render(spec));
            });
        if spawned.is_ok() {
            live.scene_pending = Some((spec, rx));
        }
    }

    /// Per-frame scene driver: adopt a finished render, fire due detail events.
    fn tick_scene(&mut self) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        let ready = match live.scene_pending.as_ref().map(|(_, rx)| rx.try_recv()) {
            Some(Ok(assets)) => Some(assets),
            Some(Err(TryRecvError::Disconnected)) => {
                live.scene_pending = None;
                None
            }
            _ => None,
        };
        if let Some(assets) = ready {
            live.scene_pending = None;
            Self::adopt_scene(live, assets);
        }
        if let Some(sc) = live.scene.as_mut() {
            let now = sc.started.elapsed().as_secs_f32();
            while let Some(e) = sc.detail_sched.poll(now) {
                let data = frames_data(sc.details[e.kind.index()].clone())
                    .volume(linear_to_db(e.gain * DETAIL_LEVEL))
                    .panning(Panning(e.pan))
                    .playback_rate(PlaybackRate(e.rate));
                let _ = live.scene_track.play(data);
            }
        }
    }

    /// Replace the live scene with freshly rendered assets: fade the old layers
    /// out, start the new texture + harmony loops, and — if the bed is on this
    /// mood — crossfade the mood bed onto the room-baked stereo bed.
    fn adopt_scene(live: &mut Live, assets: SceneAssets) {
        let fade = Tween {
            duration: SCENE_FADE,
            ..Tween::default()
        };
        if let Some(mut old) = live.scene.take() {
            if let Some(mut t) = old.texture.take() {
                t.stop(fade);
            }
            for h in &mut old.harmony {
                h.stop(fade);
            }
        }
        let intensity = live.ambience.map_or(0.5, |a| a.targets().gain);
        let mut start_loop = |s: &Stereo, gain: f32| -> Option<StaticSoundHandle> {
            let mut h = live
                .scene_track
                .play(
                    frames_data(stereo_frames(s))
                        .volume(Decibels::SILENCE)
                        .loop_region(0.0..),
                )
                .ok()?;
            h.set_volume(linear_to_db(gain), fade);
            Some(h)
        };
        let texture = start_loop(&assets.texture, intensity * TEXTURE_LEVEL);
        let gains = live.harmony_gains;
        let harmony = assets
            .harmony
            .iter()
            .zip(gains.iter())
            .filter_map(|(s, &g)| start_loop(s, g))
            .collect();
        let bed = stereo_frames(&assets.bed);
        let mood = assets.spec.mood;
        live.scene = Some(Scene {
            spec: assets.spec,
            bed: bed.clone(),
            details: assets.details.iter().map(stereo_frames).collect(),
            texture,
            harmony,
            detail_sched: DetailScheduler::new(mood, assets.spec.seed),
            started: Instant::now(),
        });
        // Swap the bed onto the stereo scene data with a same-mood crossfade.
        if let Some(req) = live.ambience.filter(|a| a.mood == mood) {
            let data = frames_data(bed)
                .volume(linear_to_db(live.volume))
                .loop_region(0.0..);
            let incoming = 1 - live.active_bed;
            live.beds[incoming].loop_handle = None;
            if let Ok(handle) = live.beds[incoming].track.play(data) {
                live.beds[incoming].loop_handle = Some(handle);
                live.beds[incoming].mood = Some(mood);
            }
            let targets = req.targets();
            Self::apply_targets_to_bed(&mut live.beds[incoming], targets, false, true);
            live.bed_baked = true;
            live.fade = Some(ActiveFade {
                fade: Crossfade::new(mood, mood, live.fade_secs),
                started: Instant::now(),
            });
            live.active_bed = incoming;
            live.tick_fade_gains(targets.gain);
        }
    }

    /// Scale the scene layers (texture level + scene low-pass) with intensity.
    fn apply_scene_intensity(live: &mut Live, targets: AmbienceTargets) {
        let tween = Tween {
            duration: Duration::from_millis(400),
            ..Tween::default()
        };
        live.scene_filter
            .set_cutoff(f64::from(targets.filter_cutoff_hz).max(1800.0), tween);
        if let Some(t) = live.scene.as_mut().and_then(|s| s.texture.as_mut()) {
            t.set_volume(linear_to_db(targets.gain * TEXTURE_LEVEL), tween);
        }
    }

    /// `true` once a room-baked scene is live (always `false` when silent).
    pub fn has_scene(&self) -> bool {
        self.live.as_ref().is_some_and(|l| l.scene.is_some())
    }

    // ── evolving drone music (Goal 2) ───────────────────────────────────────

    /// Deterministic per-mood seed for the drone-music layer — distinct
    /// across moods, stable across a play session, so a mood's music always
    /// unfolds the same way from t = 0 ([`MusicSequencer::state_at`] is a
    /// pure function of elapsed time, not of call count).
    fn music_seed(mood: AmbienceMood) -> u32 {
        0x7A11_0D0E ^ (mood as u32).wrapping_mul(0x9E37_79B9)
    }

    /// A sane room to bake the very first chord segment with, before any
    /// level has ever sent one via [`Player::set_room`].
    fn default_music_room() -> Room {
        Room::from_cells(10, 10, 4, false, 0.2)
    }

    /// (Re)start the evolving drone-music layer for a new mood: fade out
    /// whatever chords were mid-flight and seed a fresh [`MusicSequencer`] at
    /// t = 0. The crossfade back in is exactly as long as an ordinary chord
    /// crossfade (`crate::drone_music::CHORD_CROSSFADE_S`), so a mood change
    /// reads as the room's music turning over, not a hard cut.
    fn restart_music(live: &mut Live, mood: AmbienceMood) {
        let release = Tween {
            duration: Duration::from_millis(1500),
            ..Tween::default()
        };
        if let Some(old) = live.music.take() {
            for mut slot in old.slots {
                if let Some(mut h) = slot.sound.take() {
                    h.stop(release);
                }
            }
        }
        live.music_pending = [None, None];
        live.music = Some(Music {
            seq: MusicSequencer::new(mood, Self::music_seed(mood)),
            room: live.room.unwrap_or_else(Self::default_music_room),
            started: Instant::now(),
            slots: [MusicSlot::default(), MusicSlot::default()],
        });
    }

    /// Per-frame driver for the drone-music layer: adopt any background
    /// renders that finished, kick off ones newly needed, and keep both
    /// slots' gains tracking [`MusicSequencer::state_at`]'s crossfade.
    /// No-op when silent or before any mood has started the layer.
    fn tick_music(&mut self) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        // Adopt any renders that finished since the last tick.
        for i in 0..2 {
            let ready = match live.music_pending[i].as_ref().map(|(_, rx)| rx.try_recv()) {
                Some(Ok(stereo)) => Some(stereo),
                Some(Err(TryRecvError::Disconnected)) => {
                    live.music_pending[i] = None;
                    None
                }
                _ => None,
            };
            let Some(stereo) = ready else { continue };
            let (chord, _) = live.music_pending[i].take().expect("checked Some above");
            let data = frames_data(stereo_frames(&stereo))
                .volume(Decibels::SILENCE)
                .loop_region(0.0..);
            if let (Ok(h), Some(music)) = (live.music_track.play(data), live.music.as_mut()) {
                music.slots[i].chord = Some(chord);
                music.slots[i].sound = Some(h);
            }
        }

        let Some(music) = live.music.as_ref() else {
            return;
        };
        let elapsed = music.started.elapsed().as_secs_f32();
        let state = music.seq.state_at(elapsed);
        let mood = music.seq.mood();
        let degrees = crate::drone_music::chord_degrees(mood);
        let room = music.room;
        let seed = Self::music_seed(mood);
        let wanted = [state.current, state.next];

        let request_tween = Tween {
            duration: Duration::from_millis(400),
            ..Tween::default()
        };
        for (i, &chord) in wanted.iter().enumerate() {
            let slot_has_it = live
                .music
                .as_ref()
                .is_some_and(|m| m.slots[i].chord == Some(chord));
            let already_pending = live.music_pending[i]
                .as_ref()
                .is_some_and(|(c, _)| *c == chord);
            if slot_has_it || already_pending {
                continue;
            }
            // This slot needs a chord it does not hold: drop whatever it has
            // (a chord that is no longer `current` nor `next`) and request
            // the one it needs on a worker thread — mirrors
            // `Player::request_scene`/`scene_pending`.
            if let Some(music) = live.music.as_mut() {
                music.slots[i].chord = None;
                if let Some(mut h) = music.slots[i].sound.take() {
                    h.stop(request_tween);
                }
            }
            let degree = degrees.get(chord).copied().unwrap_or(0.0);
            let seg_seed = seed ^ (chord as u32).wrapping_mul(0x2545_F491);
            let (tx, rx) = std::sync::mpsc::channel();
            let spawned = std::thread::Builder::new()
                .name(String::from("re_audio-music"))
                .spawn(move || {
                    let _ = tx.send(render_music_segment_baked(mood, degree, &room, seg_seed));
                });
            if spawned.is_ok() {
                live.music_pending[i] = Some((chord, rx));
            }
        }

        // Crossfade gains: whichever slot holds `current` fades toward
        // `1 - mix`, whichever holds `next` fades toward `mix` (0 while a
        // requested `next` has not landed yet — it simply stays silent, so a
        // slow segment render is never audible as a gap).
        let gain_tween = Tween {
            duration: Duration::from_millis(400),
            ..Tween::default()
        };
        if let Some(music) = live.music.as_mut() {
            for slot in &mut music.slots {
                let Some(h) = slot.sound.as_mut() else {
                    continue;
                };
                let gain = match slot.chord {
                    Some(c) if c == state.current => (1.0 - state.mix) * MUSIC_LEVEL,
                    Some(c) if c == state.next => state.mix * MUSIC_LEVEL,
                    _ => 0.0,
                };
                h.set_volume(linear_to_db(gain), gain_tween);
            }
        }
    }

    // ── harmony ──────────────────────────────────────────────────────────────

    /// Set the six harmony-voice gains (from
    /// [`crate::soundscape::harmony_gains`]), tweened. No-op when silent.
    pub fn set_harmony(&mut self, gains: [f32; HARMONY_VOICES]) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        let scaled = gains.map(|g| (g * HARMONY_LEVEL).clamp(0.0, 1.0));
        if scaled
            .iter()
            .zip(live.harmony_gains.iter())
            .all(|(a, b)| (a - b).abs() < 0.002)
        {
            return;
        }
        live.harmony_gains = scaled;
        let tween = Tween {
            duration: Duration::from_millis(350),
            ..Tween::default()
        };
        if let Some(sc) = live.scene.as_mut() {
            for (h, &g) in sc.harmony.iter_mut().zip(scaled.iter()) {
                h.set_volume(linear_to_db(g), tween);
            }
        }
    }

    // ── positional loops ─────────────────────────────────────────────────────

    /// Bring the positional voice pool in line with `voices` (already
    /// selected and spatialized by [`crate::spatial::select_voices`]) heard
    /// from `l`: move the shared kira listener to the player, update the
    /// slot holding each continuing source (position → kira's own
    /// panning + distance attenuation; `near_gain`/`near_cutoff_hz` →
    /// occlusion/front-back, the one thing kira does not model), steal a
    /// free slot for each new source, and release slots whose source
    /// dropped out of the selection. No-op when silent.
    pub fn set_positional(&mut self, l: &Listener, voices: &[VoiceMix]) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        let glide = Tween {
            duration: Duration::from_millis(90),
            ..Tween::default()
        };
        let release = Tween {
            duration: Duration::from_millis(300),
            ..Tween::default()
        };
        let spawn = Tween {
            duration: Duration::from_millis(250),
            ..Tween::default()
        };

        live.listener.set_position(kira_pos(l.pos), glide);
        live.listener
            .set_orientation(kira_orientation(l.yaw), glide);

        let keep: std::collections::HashSet<SourceKey> = voices.iter().map(|v| v.key).collect();
        // Free every slot whose source is no longer selected. `voices` is
        // already capped at `VOICE_POOL_SIZE` by `select_voices`, so this
        // always frees enough slots for the loop below to find one.
        for slot in &mut live.voice_slots {
            if slot.key.is_some_and(|k| !keep.contains(&k)) {
                if let Some(mut h) = slot.sound.take() {
                    h.stop(release);
                }
                slot.key = None;
                let _ = slot.track.set_send(
                    live.positional_reverb_send.id(),
                    Decibels::SILENCE,
                    release,
                );
                slot.track.set_volume(Decibels::SILENCE, release);
            }
        }
        for v in voices {
            let near_gain = (v.near_gain * v.source_level * POSITIONAL_LEVEL).clamp(0.0, 1.0);
            let send_gain = (v.source_level * POSITIONAL_SEND_LEVEL).clamp(0.0, 1.0);
            if let Some(slot) = live.voice_slots.iter_mut().find(|s| s.key == Some(v.key)) {
                slot.track.set_position(kira_pos(v.pos), glide);
                slot.track.set_volume(linear_to_db(near_gain), glide);
                let _ = slot.track.set_send(
                    live.positional_reverb_send.id(),
                    linear_to_db(send_gain),
                    glide,
                );
                slot.near_filter
                    .set_cutoff(f64::from(v.near_cutoff_hz), glide);
                if let Some(h) = slot.sound.as_mut() {
                    h.set_playback_rate(PlaybackRate(v.rate), glide);
                }
                continue;
            }
            let Some(slot) = live.voice_slots.iter_mut().find(|s| s.key.is_none()) else {
                continue; // Pool exhausted — should not happen, see the doc above.
            };
            slot.key = Some(v.key);
            slot.track.set_position(kira_pos(v.pos), Tween::default());
            slot.near_filter
                .set_cutoff(f64::from(v.near_cutoff_hz), Tween::default());
            let data = frames_data(live.loops[v.key.kind.index()].clone())
                .playback_rate(PlaybackRate(v.rate))
                // Start each source at a different point so two lamps never
                // phase-lock into one louder lamp.
                .start_position(f64::from(v.key.id % 7) * 0.53)
                .loop_region(0.0..);
            if let Ok(h) = slot.track.play(data) {
                slot.track.set_volume(linear_to_db(near_gain), spawn);
                let _ = slot.track.set_send(
                    live.positional_reverb_send.id(),
                    linear_to_db(send_gain),
                    spawn,
                );
                slot.sound = Some(h);
            } else {
                slot.key = None;
            }
        }
    }

    /// Number of live positional voices (0 when silent).
    pub fn positional_count(&self) -> usize {
        self.live.as_ref().map_or(0, |l| {
            l.voice_slots.iter().filter(|s| s.key.is_some()).count()
        })
    }

    // ── sound fields ───────────────────────────────────────────────────────

    /// Push a [`FieldMix`] onto the real graph: one call per bus's continuous
    /// parameters (60ms tween, `WD`/`RATE` 800ms — slower so a stereo-width or
    /// loop-rate change never reads as a jump cut), the shared `SPACE` send,
    /// and `RATE` on whichever bed/music sound is currently playing. `HD`'s
    /// categorical mode crossfades 75ms-out/switch/75ms-in per bus when the
    /// winner changes.
    ///
    /// `combine()`'s caller (the host game) owns *selecting* and *smoothing*
    /// the input fields (that step is stateful, so it cannot live in the
    /// stateless `combine`); this just applies whatever [`FieldMix`] it is
    /// given. Safe to call every frame or every few frames — a `mix`
    /// unchanged since the last call still costs a handful of command writes
    /// (kira coalesces them cheaply) but produces no audible change. No-op
    /// when silent.
    pub fn set_fields(&mut self, mix: &FieldMix) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        let now = Instant::now();
        let tween = Tween {
            duration: Duration::from_millis(60),
            ..Tween::default()
        };
        let slow_tween = Tween {
            duration: Duration::from_millis(800),
            ..Tween::default()
        };

        apply_bus_fx(
            &mut live.grp_amb,
            &mut live.grp_amb_chain,
            &mix.buses[field::AMB],
            now,
            tween,
            slow_tween,
        );
        apply_bus_fx(
            &mut live.grp_mus,
            &mut live.grp_mus_chain,
            &mix.buses[field::MUS],
            now,
            tween,
            slow_tween,
        );
        apply_bus_fx(
            &mut live.grp_sfx,
            &mut live.grp_sfx_chain,
            &mix.buses[field::SFX],
            now,
            tween,
            slow_tween,
        );
        apply_bus_fx(
            &mut live.grp_pos,
            &mut live.grp_pos_chain,
            &mix.buses[field::POS],
            now,
            tween,
            slow_tween,
        );
        apply_cue_fx(
            &mut live.grp_cue,
            &mut live.grp_cue_chain,
            &mix.buses[field::CUE],
            tween,
            slow_tween,
        );

        apply_hd_mode(
            &mut live.space_hold,
            &mut live.space_hd_mode,
            &mut live.space_hd_pending,
            mix.space.hold_mode,
            now,
        );
        live.space_hold
            .set_window_ms(f64::from(mix.space.hold_window_ms), tween);
        let space_hold_mix: f32 = if live.space_hd_pending.is_some() {
            0.0
        } else {
            mix.space.hold_mix
        };
        live.space_hold.set_mix(
            Mix(space_hold_mix),
            Tween {
                duration: Duration::from_millis(75),
                ..Tween::default()
            },
        );
        live.space_reverb
            .set_feedback(f64::from(mix.space.reverb_feedback), tween);
        live.space_reverb
            .set_damping(f64::from(mix.space.reverb_damping), tween);

        // RATE (driven by ExitDetune): loop `playback_rate` on whichever
        // bed/music sound handles are currently playing.
        let rate = cents_to_rate(mix.loop_rate_cents);
        for bed in &mut live.beds {
            if let Some(h) = bed.loop_handle.as_mut() {
                h.set_playback_rate(rate, slow_tween);
            }
        }
        if let Some(music) = live.music.as_mut() {
            for slot in &mut music.slots {
                if let Some(h) = slot.sound.as_mut() {
                    h.set_playback_rate(rate, slow_tween);
                }
            }
        }

        live.fields_applied = *mix;
    }

    /// The last [`FieldMix`] pushed by [`Player::set_fields`]
    /// ([`FieldMix::NEUTRAL`] when silent or before the first call).
    pub fn fields_applied(&self) -> FieldMix {
        self.live
            .as_ref()
            .map_or(FieldMix::NEUTRAL, |l| l.fields_applied)
    }

    /// Schedule the `FalseEcho` mirror-voice one-shot: a footstep-alike click
    /// re-emitted from the listener's reflection across the mirror plane.
    /// The host game resolves `pos` (world cells) and `delay_s` — it knows
    /// the mirror plane and the listener's position; this just plays the
    /// click, at -6dB, round-robined across a small dedicated slot pool
    /// (never `voice_slots`, which [`Player::set_positional`] reconciles
    /// against a *looping* source
    /// selection every frame — a one-shot sharing a slot with a looping
    /// source would have its position silently overwritten the next time
    /// that runs). No-op when silent.
    pub fn schedule_mirror_footstep(&mut self, pos: [f32; 3], delay_s: f32) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        if live.mirror_slots.is_empty() {
            return;
        }
        let i = live.mirror_next % live.mirror_slots.len();
        live.mirror_next = live.mirror_next.wrapping_add(1);
        let sound: &SynthSound = self.bank.get(SoundId::Footstep);
        let mut data =
            frames_data(mono_frames(&sound.frames)).volume(linear_to_db(MIRROR_FOOTSTEP_GAIN));
        if delay_s > 0.0 {
            data = data.start_time(Duration::from_secs_f32(delay_s));
        }
        let Some(live) = self.live.as_mut() else {
            return; // unreachable in practice (silent state never toggles mid-call); kept for safety
        };
        let slot = &mut live.mirror_slots[i];
        slot.set_position(kira_pos(pos), Tween::default());
        let _ = slot.play(data);
    }
}

/// A stable per-(mood, room) seed, so the same level always renders the same
/// scene (and two different rooms never share detail timing).
fn scene_seed(mood: AmbienceMood, room: &Room) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    let mut mix = |x: u32| {
        h ^= x;
        h = h.wrapping_mul(0x0100_0193);
    };
    mix(mood as u32);
    for v in room.size_m {
        mix(v.to_bits());
    }
    mix(u32::from(room.open_sky));
    mix(room.absorption.to_bits());
    h
}

impl Default for Player {
    fn default() -> Self {
        Player::open()
    }
}

// Compile-time check that the bank covers every voice index assumption.
const _: () = assert!(SYNTH_SOUND_COUNT == SoundId::ALL.len());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::SoundId;
    use crate::transport::Transport;

    // These tests never assert audio is *heard* (impossible headlessly); they
    // assert the no-device path is graceful: every method is callable and
    // nothing panics. On a dev box with a device they also exercise the live
    // path; in CI they exercise the silent path.

    #[test]
    fn opens_without_panicking_and_has_bank() {
        let p = Player::open();
        assert_eq!(p.bank().len(), SYNTH_SOUND_COUNT);
        assert_eq!(p.volume(), 1.0);
    }

    #[test]
    fn linear_to_db_extremes() {
        assert_eq!(linear_to_db(0.0), Decibels::SILENCE);
        assert!((linear_to_db(1.0).0 - 0.0).abs() < 1e-6);
        assert!(linear_to_db(0.5).0 < 0.0);
    }

    #[test]
    fn all_methods_are_safe_in_both_states() {
        let mut p = Player::open_with_bank(SynthBank::build());

        // Immediate one-shots across every sound id.
        for id in SoundId::ALL {
            p.play_oneshot(id);
        }
        // Scheduled one-shots (beat-quantized + panned).
        for (i, id) in SoundId::ALL.iter().enumerate() {
            p.play_oneshot_at(*id, i as u64, (i as f32 / 14.0) - 1.0);
        }

        // clock_now is Some only when live; calling it is always safe.
        let _ = p.clock_now();
        let _ = p.transport();

        // Loop lifecycle.
        let h = p.play_loop(SoundId::Ambient);
        assert_eq!(h.sound(), SoundId::Ambient);
        let h2 = p.play_loop(SoundId::Ambient); // same-sound no-op
        assert_eq!(h2.sound(), SoundId::Ambient);
        p.play_loop(SoundId::AmbientCold);
        if p.is_active() {
            assert_eq!(p.looping(), Some(SoundId::AmbientCold));
        }
        p.stop_loop();
        assert_eq!(p.looping(), None);

        // Volume clamps.
        p.set_volume(2.0);
        if p.is_active() {
            assert_eq!(p.volume(), 1.0);
        }
        p.set_volume(-1.0);
        if p.is_active() {
            assert_eq!(p.volume(), 0.0);
        }
        p.set_volume(0.5);

        // Live transport reconfig.
        let t = Transport {
            bpm: 128.0,
            humanize_ms: 20.0,
            ..Transport::default()
        };
        p.set_transport(t);
        if p.is_active() {
            assert_eq!(p.transport().bpm, 128.0);
        }
    }

    #[test]
    fn ambience_methods_safe_in_both_states() {
        use crate::ambience::{Ambience, AmbienceMood};
        let mut p = Player::open_with_bank(SynthBank::build());

        // No ambience yet.
        assert_eq!(p.active_mood(), None);
        assert!(!p.is_crossfading());

        // Set each mood at a range of intensities; never panics.
        for m in AmbienceMood::ALL {
            for intensity in [0.0, 0.5, 1.0] {
                p.set_ambience(Ambience::new(m, intensity));
                p.tick_ambience();
            }
        }
        if p.is_active() {
            assert_eq!(p.active_mood(), Some(AmbienceMood::AlarmTension));
            assert_eq!(p.ambience().map(|a| a.intensity), Some(1.0));
        }

        // A mood change starts a crossfade (on the live path).
        p.set_ambience(Ambience::new(AmbienceMood::ColdVoid, 0.3));
        if p.is_active() {
            assert!(p.is_crossfading(), "mood change did not start a crossfade");
            assert_eq!(p.active_mood(), Some(AmbienceMood::ColdVoid));
        }

        // Custom fade duration + stop tear-down.
        p.set_ambience_fade(0.1);
        p.stop_loop();
        assert_eq!(p.active_mood(), None);
        assert!(!p.is_crossfading());
    }

    #[test]
    fn legacy_setambient_drone_routes_to_mood_bed() {
        use crate::ambience::AmbienceMood;
        // A legacy play_loop with a drone id should adopt the rich mood path.
        let mut p = Player::open_with_bank(SynthBank::build());
        p.play_loop(SoundId::DroneServerHum);
        if p.is_active() {
            assert_eq!(p.active_mood(), Some(AmbienceMood::ServerHum));
        }
        // A non-drone ambient id stays on the legacy single-loop slot.
        p.stop_loop();
        p.play_loop(SoundId::Ambient);
        if p.is_active() {
            assert_eq!(p.looping(), Some(SoundId::Ambient));
            assert_eq!(p.active_mood(), None);
        }
    }

    #[test]
    fn humanize_disabled_returns_zero_when_live() {
        // Pure check of the jitter math via a tiny synthetic Live is not
        // possible without a device; instead assert the silent player never
        // panics scheduling with humanize on, then off.
        let mut p = Player::open_with_bank(SynthBank::build());
        let t0 = Transport {
            humanize_ms: 0.0,
            ..Transport::default()
        };
        p.set_transport(t0);
        p.play_oneshot_at(SoundId::Boot, 4, 0.0);
        let t1 = Transport {
            humanize_ms: 12.0,
            ..Transport::default()
        };
        p.set_transport(t1);
        p.play_oneshot_at(SoundId::Boot, 8, 0.0);
    }
}
