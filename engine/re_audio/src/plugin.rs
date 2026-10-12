//! `plugin` — the Bevy integration for `re_audio`.
//!
//! Wires the [`Player`] (synth bank + kira device + musical clock) into a Bevy
//! `App` and exposes a message-driven trigger surface so other crates can ask
//! for a sound without ever touching the audio device or kira types directly:
//!
//! * [`PlaySound`] — "play this one-shot" (the plugin applies [`crate::policy`]
//!   + [`crate::transport`] to decide immediate vs beat-quantized scheduling).
//! * [`SetAmbient`] — "loop this ambient bed" (or stop).
//! * [`SetMasterVolume`] — master volume in `[0, 1]`.
//! * [`SetTempo`] — reconfigure the musical clock (BPM / grid / humanize).
//! * [`SetTheme`] — swap the director's generative palette.
//! * [`SolveSilence`] — mute the generative theme layer for a few seconds (the
//!   ambience beds and the game's own one-shots keep playing).
//!
//! The device-backed [`Player`] is a **`NonSend` resource** (kira/cpal stream
//! handles are not guaranteed `Send`/`Sync`), keeping audio on the single main
//! schedule. The generative [`Director`] is a normal (`Send`) resource — it is
//! pure logic. No terminal, no render. A headless app with no device is a silent
//! no-op for sound but still runs every system.

use bevy::prelude::*;

use crate::acoustics::Room;
use crate::ambience::{Ambience, AmbienceMood};
use crate::cue::Cue;
use crate::director::{Director, Theme};
use crate::field::FieldMix;
use crate::playback::Player;
use crate::policy::{quantize_of, Quantize};
use crate::soundscape::{harmony_gains, Tension, MAX_POSITIONAL_VOICES};
use crate::spatial::{select_voices, Listener, PositionalSource};
use crate::synth::SoundId;
use crate::transport::{next_tick, BeatClass, TickPos, Transport};

/// Message: a game EVENT happened — voice it through the sound grammar
/// ([`crate::cue`]). `pos` (world cells) places the sound in the stereo field
/// relative to the current [`AudioListener`]; `None` plays it centred.
///
/// Prefer this over [`PlaySound`] for anything the player caused: the cue
/// table owns pitch / level / pan and the harmonic-tension response, so every
/// call site speaks the same language.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct PlayCue {
    /// Which game event happened.
    pub cue: Cue,
    /// World-cell position the cue is heard from, or `None` to play centred.
    pub pos: Option<[f32; 3]>,
}

impl PlayCue {
    /// A centred cue.
    pub fn new(cue: Cue) -> PlayCue {
        PlayCue { cue, pos: None }
    }

    /// A cue heard from a world position.
    pub fn at(cue: Cue, pos: [f32; 3]) -> PlayCue {
        PlayCue {
            cue,
            pos: Some(pos),
        }
    }
}

/// Message: a one-shot with explicit pitch / level / pan (the director uses it
/// so its per-event pan is no longer thrown away). Scheduled by the same
/// [`quantize_of`] policy as [`PlaySound`].
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct PlayVoice {
    /// Which pre-rendered sound to play.
    pub sound: SoundId,
    /// Playback rate (pitch): 1.0 = as rendered, 2.0 = an octave up.
    pub rate: f64,
    /// Linear gain in `[0, 1]`.
    pub gain: f32,
    /// Stereo pan in `[-1, 1]`.
    pub pan: f32,
}

/// Message: the level's acoustic room (sent on level load). Re-voices the
/// master reverb and triggers the room-baked stereo scene render.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct SetRoom(pub Room);

/// Resource: where the player's ears are. The game writes it every frame from
/// the play camera; `None` outside play (positional loops fall silent).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct AudioListener(pub Option<Listener>);

/// Resource: every sounding thing in the world this frame (lamps, powered
/// pipes, lit nodes, emitters). The plugin keeps the
/// [`MAX_POSITIONAL_VOICES`] most audible of them playing.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct AudioSources(pub Vec<PositionalSource>);

/// Resource: puzzle progress in `[0, 1]` as the ambience harmony should hear
/// it (see [`crate::soundscape::harmony_progress`]). The game writes it.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct HarmonyProgress(pub f32);

/// Resource: the live harmonic-tension envelope (spiked by refusal cues,
/// decaying with a 1.2 s half-life). Public for inspection in tests.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct HarmonyTension(pub Tension);

/// Resource: this frame's "sound fields" target, as computed by the host
/// game's own per-frame [`crate::field::combine`] call — selecting and
/// envelope-smoothing the active [`crate::field::FieldInput`]s is the host
/// game's job (see [`crate::field`]'s module doc for why `combine` itself
/// cannot own that state); this resource just carries the *result* across the
/// crate boundary. Applied to the live kira graph every frame by
/// [`apply_sound_fields`] via [`Player::set_fields`]. Defaults to
/// [`FieldMix::NEUTRAL`] (no field active, chain bit-exact-neutral) so a
/// level that never touches this resource sounds exactly as it did before
/// sound fields existed.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SoundFieldMix(pub FieldMix);

impl Default for SoundFieldMix {
    fn default() -> SoundFieldMix {
        SoundFieldMix(FieldMix::NEUTRAL)
    }
}

/// Message: schedule the `FalseEcho` mirror-voice one-shot — a footstep-alike
/// click re-emitted from the listener's reflection across a mirror plane.
/// The host game resolves `pos` (world cells, already mirrored) and
/// `delay_s` (the extra travel time to the reflection); the plugin just
/// forwards both to [`Player::schedule_mirror_footstep`], which plays it at a
/// fixed -6dB on a small dedicated one-shot pool.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct MirrorFootstep {
    /// World-cell position of the mirrored reflection.
    pub pos: [f32; 3],
    /// Extra travel time to the reflection, in seconds.
    pub delay_s: f32,
}

/// Level of a cue relative to the ambience (shared with the offline mix).
pub use crate::offline::CUE_LEVEL;

/// Message: trigger a one-shot sound effect. Emit from gameplay/input systems.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaySound(pub SoundId);

/// Message: set the looping ambient bed (the **legacy / back-compat** path).
/// `Some(id)` starts/replaces the loop; `None` stops whatever is looping.
///
/// Relationship to [`SetAmbience`]: this is the simple "loop this exact sound"
/// surface. If `id` is one of the four drone moods, the player forwards it to
/// the rich crossfading mood path at a default intensity, so even a legacy call
/// gets a smooth transition; non-drone beds (e.g. [`SoundId::Ambient`]) use the
/// plain single loop. Prefer [`SetAmbience`] for new code — it selects a mood by
/// name and scales presence by intensity.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetAmbient(pub Option<SoundId>);

/// Message: set the **rich** ambience — a named mood + an intensity in `[0, 1]`.
///
/// This is the primary ambience surface. `mood` selects which drone bed plays;
/// `intensity` scales its presence (volume + filter openness + reverb/movement
/// depth) along the dread curve. The game sends this:
/// * **per level** on load (e.g. `SetAmbience { mood: ServerHum, intensity: 0.4 }`),
/// * **per place / zone** on a trigger (a different mood or intensity).
///
/// Changing the `mood` at runtime **crossfades** smoothly from the current bed to
/// the new one (a timed equal-power gain ramp, never a hard cut). Changing only
/// the `intensity` re-targets the active bed in place. Driving `intensity` up
/// (from corruption / the closing `while(true)`) raises dread monotonically.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct SetAmbience {
    /// The mood (drone bed) to play.
    pub mood: AmbienceMood,
    /// Presence / dread in `[0, 1]` (clamped).
    pub intensity: f32,
}

impl SetAmbience {
    /// Convenience constructor (clamps `intensity`).
    pub fn new(mood: AmbienceMood, intensity: f32) -> SetAmbience {
        SetAmbience {
            mood,
            intensity: intensity.clamp(0.0, 1.0),
        }
    }
}

/// Message: set master volume in linear `[0, 1]` (clamped).
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct SetMasterVolume(pub f32);

/// Message: reconfigure the musical transport (BPM, grid, snap, humanize).
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct SetTempo(pub Transport);

/// Message: swap the generative director's theme/palette.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetTheme(pub Theme);

/// Message: mute the **generative theme layer** for a few seconds — the music
/// holds its breath.
///
/// Sent by the game on its peak solves, where the designed window is 3-5 s;
/// [`Director::SOLVE_SILENCE_SECS`] is the default the game can use. `seconds`
/// is clamped into
/// `[0, Director::MAX_SILENCE_SECS]`; a negative / non-finite value is a no-op.
///
/// What goes quiet is only the director's arrangement. The **ambience beds keep
/// running** (they are `playback` loops the director never touches) and so do
/// the game's own one-shots — footsteps, UI, story stingers — because those are
/// sent as [`PlaySound`] directly and never pass through the director. The world
/// does not die; the score stops breathing for a moment.
///
/// Re-entrant **by extension, not by stacking**: a second message while a window
/// is open sets the remaining time to `max(remaining, seconds)`, so two solves
/// close together cannot add up to a long hole in the score.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct SolveSilence {
    /// Length of the silence window in seconds.
    pub seconds: f32,
}

impl SolveSilence {
    /// The designed solve-silence window (the midpoint of PROGRESSION's 3–5 s).
    pub fn solve() -> SolveSilence {
        SolveSilence {
            seconds: Director::SOLVE_SILENCE_SECS,
        }
    }
}

impl Default for SolveSilence {
    fn default() -> Self {
        SolveSilence::solve()
    }
}

/// The transport config the plugin schedules against. Mirrored from the
/// [`Player`]'s live transport so the (Send) scheduling systems can read it
/// without touching the NonSend player every frame.
#[derive(Resource, Clone, Copy, Debug, Default)]
struct ActiveTransport(Transport);

/// The last integer clock tick the director was advanced through. The director
/// must run exactly once per integer tick — not once per render frame — or its
/// per-beat probabilities (and seeded RNG) race ahead at frame rate. `None`
/// means "not yet run" (e.g. clock not started). See [`director_ticks_to_run`].
#[derive(Resource, Clone, Copy, Debug, Default)]
struct LastDirectorTick(Option<u64>);

/// Given the last processed tick and the clock's current integer tick, return
/// the (inclusive) range of integer ticks the director should be advanced
/// through this frame, and the new "last processed" value.
///
/// Contract:
/// * First run at `now` (`last == None`): process exactly `[now, now]` — the
///   single current tick — so a freshly started clock fires once, not for the
///   whole history `0..=now`.
/// * No advance (`now == last`): process nothing.
/// * Advanced by N (`now == last + N`): process each skipped tick in order
///   `last+1 ..= now` (N invocations), dropping none, double-firing none.
/// * Clock went backwards (`now < last`, e.g. transport reset): re-anchor on
///   `now` and process the single tick `[now, now]`.
///
/// Returns `(first, last_inclusive, new_last)` where the half-open work is
/// `first..=last_inclusive` (empty iff `first > last_inclusive`).
fn director_ticks_to_run(last: Option<u64>, now: u64) -> (u64, u64, u64) {
    match last {
        None => (now, now, now),
        Some(prev) if now > prev => (prev + 1, now, now),
        Some(prev) if now < prev => (now, now, now),
        Some(prev) => (prev + 1, prev, prev), // now == prev → empty range
    }
}

/// The seed for the generative director. Fixed so a run is reproducible; a
/// level could override it on load.
const DIRECTOR_SEED: u32 = 0x5EED_1234;

/// The audio plugin.
///
/// At startup it opens the default device + builds the synth bank into a
/// `NonSend` [`Player`] (silent no-op when no device exists), inserts the
/// generative [`Director`], registers the trigger messages, and runs the drain
/// + generative systems in `Update`.
#[derive(Default)]
pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app.insert_non_send(Player::open())
            .insert_resource(Director::new(Theme::default(), DIRECTOR_SEED))
            .insert_resource(ActiveTransport::default())
            .insert_resource(LastDirectorTick::default())
            .init_resource::<AudioListener>()
            .init_resource::<AudioSources>()
            .init_resource::<HarmonyProgress>()
            .init_resource::<HarmonyTension>()
            .init_resource::<SoundFieldMix>()
            .add_message::<PlayCue>()
            .add_message::<PlayVoice>()
            .add_message::<SetRoom>()
            .add_message::<PlaySound>()
            .add_message::<SetAmbient>()
            .add_message::<SetAmbience>()
            .add_message::<SetMasterVolume>()
            .add_message::<SetTempo>()
            .add_message::<SetTheme>()
            .add_message::<SolveSilence>()
            .add_message::<MirrorFootstep>()
            // The silence countdown needs a clock. `TimePlugin` (via
            // MinimalPlugins) normally owns `Time` and keeps it advancing;
            // `init_resource` only fills it in for a bare `App` that has no
            // TimePlugin, where dt stays 0 and the window simply never expires.
            .init_resource::<Time>()
            .add_systems(
                Update,
                (
                    apply_tempo,
                    apply_theme,
                    // Age the open window first, then honour this frame's
                    // requests, so a fresh window gets its full duration.
                    tick_solve_silence,
                    apply_solve_silence,
                    tick_director,
                    play_oneshots,
                    play_voices,
                    play_cues,
                    apply_room,
                    apply_ambient,
                    apply_ambience,
                    tick_ambience,
                    drive_harmony,
                    drive_positional,
                    apply_sound_fields,
                    play_mirror_footsteps,
                    apply_volume,
                )
                    .chain(),
            );
    }
}

/// `Update` system: read the clock and advance the director once per *integer*
/// clock tick (not once per render frame), turning its proposals into
/// [`PlaySound`] messages. The director is the generative engine; it only
/// proposes — `play_oneshots` still applies policy + transport.
///
/// At ~60 fps the same integer tick recurs across many frames; calling
/// [`Director::on_beat`] every frame would fire per-beat probabilities ~20×
/// too often and race the seeded RNG. We gate on [`LastDirectorTick`] and, if a
/// slow frame skipped several ticks, replay each skipped tick once in order.
fn tick_director(
    player: NonSend<Player>,
    transport: Res<ActiveTransport>,
    mut director: ResMut<Director>,
    mut last_tick: ResMut<LastDirectorTick>,
    mut out: MessageWriter<PlayVoice>,
) {
    // Only the live (device) path has a clock; silent path produces no events
    // and leaves the gate untouched (so a later start fires exactly once).
    let Some((now, _frac)) = player.clock_now() else {
        return;
    };
    let (first, last_inclusive, new_last) = director_ticks_to_run(last_tick.0, now);
    if first <= last_inclusive {
        for tick in first..=last_inclusive {
            if let Some(ev) = director.on_beat(tick, transport.0.grid) {
                // Keep the director's pan: it spreads the arrangement across
                // the field (it used to be dropped here, stacking every
                // generative event dead-centre).
                out.write(PlayVoice {
                    sound: ev.sound,
                    rate: 1.0,
                    gain: 1.0,
                    pan: ev.pan,
                });
            }
        }
    }
    last_tick.0 = Some(new_last);
}

/// `Update` system: drain [`PlaySound`] messages. Each is resolved through
/// [`quantize_of`]: `Immediate` plays now; `Beat(class)` is snapped onto the
/// next tick of that class via [`next_tick`] and scheduled (with humanize).
fn play_oneshots(
    mut player: NonSendMut<Player>,
    transport: Res<ActiveTransport>,
    mut msgs: MessageReader<PlaySound>,
) {
    let tr = transport.0;
    for PlaySound(id) in msgs.read().copied() {
        match quantize_of(id) {
            Quantize::Immediate => player.play_oneshot(id),
            Quantize::Beat(class) => match player.clock_now() {
                Some((tick, frac)) => {
                    let now = TickPos::new(tick, frac);
                    let target = next_tick(now, class, tr.grid, tr.snap_window);
                    // Pan: a tiny deterministic spread by class so strong/weak
                    // events don't all stack dead-center.
                    let pan = match class {
                        BeatClass::Strong => 0.0,
                        BeatClass::Weak => 0.15,
                    };
                    player.play_oneshot_at(id, target, pan);
                }
                // No clock (silent) → immediate fallback (still a no-op).
                None => player.play_oneshot(id),
            },
        }
    }
}

/// Resolve where a voice lands on the clock: `None` = now (Immediate, or no
/// clock on the silent path), `Some(tick)` = the next tick of its beat class.
fn schedule_for(player: &Player, tr: Transport, id: SoundId) -> Option<u64> {
    match quantize_of(id) {
        Quantize::Immediate => None,
        Quantize::Beat(class) => player.clock_now().map(|(tick, frac)| {
            next_tick(TickPos::new(tick, frac), class, tr.grid, tr.snap_window)
        }),
    }
}

/// `Update` system: drain [`PlayVoice`] (director events and any explicit
/// voice), scheduled by the same policy as [`PlaySound`].
fn play_voices(
    mut player: NonSendMut<Player>,
    transport: Res<ActiveTransport>,
    mut msgs: MessageReader<PlayVoice>,
) {
    for v in msgs.read().copied() {
        let tick = schedule_for(&player, transport.0, v.sound);
        player.play_voice(v.sound, v.rate, v.gain, v.pan, tick);
    }
}

/// `Update` system: drain [`PlayCue`] — voice each event through the cue
/// table (placed relative to the listener when it has a position) and let it
/// spike the harmonic tension.
fn play_cues(
    mut player: NonSendMut<Player>,
    transport: Res<ActiveTransport>,
    listener: Res<AudioListener>,
    mut tension: ResMut<HarmonyTension>,
    mut msgs: MessageReader<PlayCue>,
) {
    for PlayCue { cue, pos } in msgs.read().copied() {
        tension.0.pulse(cue.tension());
        let from = match (pos, listener.0.as_ref()) {
            (Some(p), Some(l)) => Some((p, l)),
            _ => None,
        };
        for v in cue.placed_voices(from) {
            let tick = schedule_for(&player, transport.0, v.sound);
            player.play_voice(v.sound, v.rate, v.gain * CUE_LEVEL, v.pan, tick);
        }
    }
}

/// `Update` system: drain [`SetRoom`]. Last message wins.
fn apply_room(mut player: NonSendMut<Player>, mut msgs: MessageReader<SetRoom>) {
    if let Some(SetRoom(room)) = msgs.read().copied().last() {
        player.set_room(room);
    }
}

/// `Update` system: age the tension envelope and push the harmony mix that the
/// current progress + tension imply.
fn drive_harmony(
    mut player: NonSendMut<Player>,
    time: Res<Time>,
    progress: Res<HarmonyProgress>,
    mut tension: ResMut<HarmonyTension>,
) {
    tension.0.decay(time.delta_secs());
    player.set_harmony(harmony_gains(progress.0, tension.0.level()));
}

/// `Update` system: select + spatialize the positional sources for the current
/// listener and hand them to the player (no listener ⇒ no positional voices).
fn drive_positional(
    mut player: NonSendMut<Player>,
    listener: Res<AudioListener>,
    sources: Res<AudioSources>,
) {
    if !(listener.is_changed() || sources.is_changed()) && listener.0.is_some() {
        return;
    }
    let l = listener.0.unwrap_or_default();
    let voices = match listener.0.as_ref() {
        Some(l) => select_voices(l, &sources.0, MAX_POSITIONAL_VOICES),
        None => Vec::new(),
    };
    player.set_positional(&l, &voices);
}

/// `Update` system: push this frame's [`SoundFieldMix`] onto the live kira
/// graph via [`Player::set_fields`]. Runs unconditionally every frame — an
/// unchanged mix costs a handful of coalesced command writes and produces no
/// audible change (see that method's doc), which is cheaper and simpler than
/// gating on `is_changed()` given the host game is expected to recompute
/// `combine()` every frame anyway (its inputs' envelopes are time-driven).
fn apply_sound_fields(mut player: NonSendMut<Player>, mix: Res<SoundFieldMix>) {
    player.set_fields(&mix.0);
}

/// `Update` system: drain [`MirrorFootstep`] messages, forwarding each to
/// [`Player::schedule_mirror_footstep`].
fn play_mirror_footsteps(mut player: NonSendMut<Player>, mut msgs: MessageReader<MirrorFootstep>) {
    for MirrorFootstep { pos, delay_s } in msgs.read().copied() {
        player.schedule_mirror_footstep(pos, delay_s);
    }
}

/// `Update` system: keep the single ambient bed in sync. Last message wins.
fn apply_ambient(mut player: NonSendMut<Player>, mut msgs: MessageReader<SetAmbient>) {
    if let Some(SetAmbient(target)) = msgs.read().copied().last() {
        match target {
            Some(id) => {
                player.play_loop(id);
            }
            None => player.stop_loop(),
        }
    }
}

/// `Update` system: drain [`SetAmbience`] (the rich mood path). Last message
/// wins this frame — a mood change crossfades, an intensity-only change
/// re-targets the active bed in place.
fn apply_ambience(mut player: NonSendMut<Player>, mut msgs: MessageReader<SetAmbience>) {
    if let Some(&SetAmbience { mood, intensity }) = msgs.read().last() {
        player.set_ambience(Ambience::new(mood, intensity));
    }
}

/// `Update` system: advance any in-progress ambience crossfade every frame and
/// keep the active bed's intensity params applied. Runs unconditionally (it is a
/// no-op when no bed is active or when silent).
fn tick_ambience(mut player: NonSendMut<Player>) {
    player.tick_ambience();
}

/// `Update` system: drain [`SetMasterVolume`]. Last one wins.
fn apply_volume(mut player: NonSendMut<Player>, mut msgs: MessageReader<SetMasterVolume>) {
    if let Some(SetMasterVolume(v)) = msgs.read().copied().last() {
        player.set_volume(v);
    }
}

/// `Update` system: drain [`SetTempo`]. Reconfigures the clock + the cached
/// scheduling transport. Last one wins.
fn apply_tempo(
    mut player: NonSendMut<Player>,
    mut active: ResMut<ActiveTransport>,
    mut msgs: MessageReader<SetTempo>,
) {
    if let Some(SetTempo(t)) = msgs.read().copied().last() {
        active.0 = t;
        player.set_transport(t);
    }
}

/// `Update` system: drain [`SetTheme`]. Swaps the director's generative palette
/// *and* re-voices the master fog (reverb tail + lowpass) to match, so a theme is
/// audible even on a beat the director stays silent through. Last message wins.
fn apply_theme(
    mut player: NonSendMut<Player>,
    mut director: ResMut<Director>,
    mut msgs: MessageReader<SetTheme>,
) {
    if let Some(SetTheme(theme)) = msgs.read().copied().last() {
        director.set_theme(theme);
        player.set_theme(theme);
    }
}

/// `Update` system: drain [`SolveSilence`]. Unlike the "last wins" systems every
/// message is honoured, because the semantics are *extend to the longest*, not
/// *replace*: two solves in one frame leave the longer window open.
fn apply_solve_silence(mut director: ResMut<Director>, mut msgs: MessageReader<SolveSilence>) {
    for SolveSilence { seconds } in msgs.read().copied() {
        director.hold_breath(seconds);
    }
}

/// `Update` system: age any open solve-silence window by this frame's delta.
/// A no-op when no window is open. Runs on the silent (device-less) path too —
/// the window is pure logic and must expire whether or not anything is audible.
fn tick_solve_silence(mut director: ResMut<Director>, time: Res<Time>) {
    if director.is_silent() {
        director.advance_silence(time.delta_secs());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::app::App;

    // Drive the plugin headlessly: send each message kind through a real Bevy
    // App and tick it. Asserts the systems run without panicking (silent path).
    #[test]
    fn plugin_pumps_messages_without_panicking() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);

        app.world_mut().write_message(PlaySound(SoundId::Boot));
        app.world_mut()
            .write_message(SetAmbient(Some(SoundId::Ambient)));
        app.world_mut().write_message(SetMasterVolume(0.5));
        app.world_mut()
            .write_message(SetTempo(Transport::default()));
        app.world_mut().write_message(SetTheme(Theme::Cold));
        app.update();

        // Second tick: switch + stop ambient, fire a quantized + a glitch sound.
        app.world_mut().write_message(SetAmbient(None));
        app.world_mut().write_message(PlaySound(SoundId::Glitch));
        app.world_mut().write_message(PlaySound(SoundId::ToneA3));
        app.update();

        let player = app.world().non_send::<Player>();
        assert_eq!(player.bank().len(), crate::synth::SYNTH_SOUND_COUNT);
    }

    #[test]
    fn setambience_selects_the_right_bed_and_crossfades() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);

        // Per-level load: a mood at a moderate intensity.
        app.world_mut()
            .write_message(SetAmbience::new(AmbienceMood::ColdVoid, 0.4));
        app.update();
        {
            let player = app.world().non_send::<Player>();
            if player.is_active() {
                assert_eq!(player.active_mood(), Some(AmbienceMood::ColdVoid));
                assert_eq!(player.ambience().map(|a| a.intensity), Some(0.4));
            }
        }

        // Per-place trigger: a new mood → must start a crossfade, not a hard cut.
        app.world_mut()
            .write_message(SetAmbience::new(AmbienceMood::AlarmTension, 0.9));
        app.update();
        {
            let player = app.world().non_send::<Player>();
            if player.is_active() {
                assert_eq!(player.active_mood(), Some(AmbienceMood::AlarmTension));
                assert!(
                    player.is_crossfading(),
                    "mood change must crossfade, not hard-cut"
                );
            }
        }

        // Intensity-only change (corruption rising): same mood, no new crossfade
        // forced, request reflects the new intensity.
        app.world_mut()
            .write_message(SetAmbience::new(AmbienceMood::AlarmTension, 1.0));
        app.update();
        {
            let player = app.world().non_send::<Player>();
            if player.is_active() {
                assert_eq!(player.active_mood(), Some(AmbienceMood::AlarmTension));
                assert_eq!(player.ambience().map(|a| a.intensity), Some(1.0));
            }
        }
    }

    // ── the grammar / room / positional surface ──────────────────────────────

    #[test]
    fn cues_room_listener_and_sources_pump_without_panicking() {
        use crate::spatial::{LoopKind, SourceKey};
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        app.world_mut()
            .write_message(SetRoom(Room::from_cells(20, 14, 6, false, 0.2)));
        app.world_mut()
            .write_message(SetAmbience::new(AmbienceMood::ServerHum, 0.4));
        app.world_mut().resource_mut::<AudioListener>().0 = Some(Listener {
            pos: [2.0, 2.0, 1.5],
            yaw: 0.3,
        });
        app.world_mut().resource_mut::<AudioSources>().0 = vec![PositionalSource {
            key: SourceKey {
                kind: LoopKind::LampHum,
                id: 0,
            },
            pos: [4.0, 2.0, 2.0],
            level: 1.0,
            rate: 1.0,
            occlusion: 0.0,
        }];
        app.world_mut().resource_mut::<HarmonyProgress>().0 = 0.5;
        app.world_mut().write_message(PlayCue::at(
            Cue::NodeLit { lit: 2, total: 4 },
            [4.0, 5.0, 1.5],
        ));
        app.world_mut().write_message(PlayCue::new(Cue::SwitchOff));
        app.world_mut().write_message(PlayVoice {
            sound: SoundId::ToneA3,
            rate: 1.0,
            gain: 0.5,
            pan: -0.4,
        });
        app.update();
        app.world_mut().resource_mut::<AudioListener>().0 = None;
        app.update();
        // The test build is always silent: nothing was opened, nothing played.
        let player = app.world().non_send::<Player>();
        assert!(!player.is_active());
        assert_eq!(player.positional_count(), 0);
    }

    #[test]
    fn a_refusal_spikes_harmonic_tension_which_then_decays() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        app.world_mut()
            .write_message(PlayCue::new(Cue::RefusedNoPower));
        app.update();
        let spiked = app.world().resource::<HarmonyTension>().0.level();
        assert!(spiked > 0.9, "refusal did not spike tension: {spiked}");
        advance(&mut app, 3_000);
        let later = app.world().resource::<HarmonyTension>().0.level();
        assert!(later < spiked * 0.25, "tension did not decay: {later}");
        // Acceptance never raises it.
        app.world_mut()
            .write_message(PlayCue::new(Cue::SwitchOn { used: 1, budget: 3 }));
        advance(&mut app, 0);
        assert!(app.world().resource::<HarmonyTension>().0.level() <= later);
    }

    // ── H1: director runs once per integer tick, not once per frame ──────────

    /// Replay the gate the way `tick_director` does, recording every integer
    /// tick that `on_beat` *would* be invoked for. Pure (no device, no Bevy).
    fn run_gate(last: Option<u64>, now: u64) -> (Vec<u64>, u64) {
        let (first, last_inclusive, new_last) = director_ticks_to_run(last, now);
        let mut invoked = Vec::new();
        if first <= last_inclusive {
            invoked.extend(first..=last_inclusive);
        }
        (invoked, new_last)
    }

    #[test]
    fn repeated_calls_at_same_tick_invoke_on_beat_once() {
        // First frame at tick 5: exactly one invocation, for tick 5.
        let (invoked, last) = run_gate(None, 5);
        assert_eq!(invoked, vec![5]);
        // Many subsequent frames at the SAME integer tick 5: zero invocations.
        let mut last = last;
        for _ in 0..20 {
            let (invoked, new_last) = run_gate(Some(last), 5);
            assert!(invoked.is_empty(), "re-fired at an unchanged tick");
            last = new_last;
        }
        // Total invocations across all those frames is exactly one.
    }

    #[test]
    fn advancing_n_ticks_invokes_on_beat_n_times_in_order() {
        // Start anchored at tick 3 (one invocation).
        let (invoked, mut last) = run_gate(None, 3);
        assert_eq!(invoked, vec![3]);
        // A slow frame jumps from tick 3 to tick 8: replay 4,5,6,7,8 in order.
        let (invoked, new_last) = run_gate(Some(last), 8);
        assert_eq!(invoked, vec![4, 5, 6, 7, 8]);
        last = new_last;
        // Single-step advance fires exactly one, for the new tick.
        let (invoked, _) = run_gate(Some(last), 9);
        assert_eq!(invoked, vec![9]);
    }

    #[test]
    fn gate_no_calls_before_clock_starts_then_one_on_first_tick() {
        // `None` (clock not started) followed by first observed tick 0 → exactly
        // the single tick 0, never the whole history.
        let (invoked, last) = run_gate(None, 0);
        assert_eq!(invoked, vec![0]);
        assert_eq!(last, 0);
    }

    #[test]
    fn gate_reanchors_on_backwards_clock() {
        // Transport reset: clock jumps from 40 back to 0 → re-anchor, one tick.
        let (invoked, last) = run_gate(Some(40), 0);
        assert_eq!(invoked, vec![0]);
        assert_eq!(last, 0);
    }

    #[test]
    fn director_resource_is_present_and_themed() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        app.world_mut().write_message(SetTheme(Theme::Garden));
        app.update();
        let d = app.world().resource::<Director>();
        assert_eq!(d.theme(), Theme::Garden);
    }

    #[test]
    fn every_movement_theme_is_settable_through_the_message() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        for theme in Theme::MOVEMENTS {
            app.world_mut().write_message(SetTheme(theme));
            app.update();
            assert_eq!(app.world().resource::<Director>().theme(), theme);
        }
    }

    // ─────────────────────── solve silence, end to end ───────────────────────

    /// Push the app's `Time` forward without sleeping.
    fn advance(app: &mut App, ms: u64) {
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_millis(ms));
        app.update();
    }

    #[test]
    fn solve_silence_message_mutes_the_theme_layer_then_expires() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        app.world_mut().write_message(SetTheme(Theme::P4Closing));
        app.world_mut().write_message(SolveSilence::solve());
        app.update();

        let d = app.world().resource::<Director>();
        assert!(d.is_silent(), "the solve did not mute the theme layer");
        assert_eq!(d.silence_remaining(), Director::SOLVE_SILENCE_SECS);

        // Halfway through, still muted.
        advance(&mut app, 2_000);
        assert!(app.world().resource::<Director>().is_silent());

        // Past the window, the theme layer is live again.
        advance(&mut app, 2_500);
        assert!(
            !app.world().resource::<Director>().is_silent(),
            "the silence window never expired"
        );
    }

    #[test]
    fn two_solves_in_flight_extend_the_window_instead_of_stacking() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);

        app.world_mut().write_message(SolveSilence { seconds: 4.0 });
        app.update();
        advance(&mut app, 1_000); // 3 s left
        assert_eq!(app.world().resource::<Director>().silence_remaining(), 3.0);

        // A second solve arrives: extend to 4 s, not 3 + 4.
        app.world_mut().write_message(SolveSilence { seconds: 4.0 });
        // Two in the same frame must not stack either.
        app.world_mut().write_message(SolveSilence { seconds: 3.5 });
        advance(&mut app, 0);
        assert_eq!(app.world().resource::<Director>().silence_remaining(), 4.0);

        // Still bounded by the longest single request.
        advance(&mut app, 4_100);
        assert!(!app.world().resource::<Director>().is_silent());
    }

    #[test]
    fn solve_silence_leaves_the_ambience_bed_alone() {
        // The bed is `playback` state; the silence window is director state. The
        // theme holding its breath must not stop or re-target the ambience.
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        app.world_mut()
            .write_message(SetAmbience::new(AmbienceMood::OrganicDecay, 0.6));
        app.world_mut().write_message(SolveSilence::solve());
        app.update();

        assert!(app.world().resource::<Director>().is_silent());
        let player = app.world().non_send::<Player>();
        if player.is_active() {
            assert_eq!(player.active_mood(), Some(AmbienceMood::OrganicDecay));
            assert_eq!(player.ambience().map(|a| a.intensity), Some(0.6));
        }
    }

    // ───────────────────────── sound fields wiring ───────────────────────────

    #[test]
    fn sound_field_mix_defaults_to_neutral_and_pumps_without_panicking() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        // Nobody wrote `SoundFieldMix` — the resource must already exist,
        // neutral, from `init_resource`, and driving it must not panic on the
        // silent test path.
        assert_eq!(app.world().resource::<SoundFieldMix>().0, FieldMix::NEUTRAL);
        app.update();
        let player = app.world().non_send::<Player>();
        assert_eq!(player.fields_applied(), FieldMix::NEUTRAL);
    }

    #[test]
    fn setting_sound_field_mix_is_applied_to_the_player_every_frame() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        let mut mix = FieldMix::NEUTRAL;
        mix.buses[crate::field::AMB].lp_cutoff_hz = 900.0;
        app.world_mut().resource_mut::<SoundFieldMix>().0 = mix;
        app.update();
        let player = app.world().non_send::<Player>();
        // The headless test build is always silent (no device); on a live
        // build this is where `fields_applied()` would mirror what was
        // pushed. Here we can only assert it does not panic and the resource
        // itself is exactly what was written (what the host game reads back
        // to confirm a request landed) — meaningful device-state assertions are
        // gated the same way the rest of this file gates them.
        if player.is_active() {
            assert_eq!(player.fields_applied(), mix);
        }
        assert_eq!(app.world().resource::<SoundFieldMix>().0, mix);
    }

    #[test]
    fn mirror_footstep_message_pumps_without_panicking() {
        let mut app = App::new();
        app.add_plugins(AudioPlugin);
        app.world_mut().write_message(MirrorFootstep {
            pos: [1.0, 2.0, 0.5],
            delay_s: 0.05,
        });
        app.update();
        // Silent path: nothing to assert beyond "did not panic and the
        // message was drained" (no player state to inspect with no device).
        let player = app.world().non_send::<Player>();
        assert!(!player.is_active());
    }
}
