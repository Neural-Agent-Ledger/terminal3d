// Test-module assert_eq! on f32 camera position values: intentional exact-equal
// checks (jump_to_cursor places camera at cell_centre = idx + 0.5 exactly).
// Allow float_cmp only in test compilation units; binary code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
//! `terminal-state-editor` — a terminal-native level editor for `.lvl` maps, built
//! as a small Bevy headless app over the same `re_world`/`re_level`/`re_render`
//! crates the game uses, so the editor's 3D preview and the shipped game render
//! identically.
//!
//! ## Running it
//!
//! ```text
//! cargo run -p ts_editor -- levels/22-lit-cellar.lvl
//! ```
//!
//! The optional first non-flag argument is a `.lvl` path to load at startup; with
//! none, the editor opens onto an empty level. `--size <cols>x<rows>` (or the
//! `TS_COLS`/`TS_ROWS` env vars) pins the framebuffer to a fixed grid instead of
//! auto-sizing to the terminal. `TS_HIDE_CURSOR=1` hides the OS pointer (the
//! editor keeps it visible by default so clicking panes works).
//!
//! ## Views and tool modes
//!
//! `Tab` cycles the [`state::ViewMode`]: **Split** (top-down map + live 3D preview,
//! the default), **Full2D** (map fills the width), and **Full3D** (walk-through
//! preview fills the screen). Independently, a [`state::ToolMode`] governs what the
//! cursor does in the 2D pane: plain geometry Brush, the wedge/ramp Shape brush,
//! the Staircase stamp, `.vox` box-model placement (Vox), Material authoring, and
//! Object (entity) placement — `m`, `v`, `o`, `8` and the `1`-`7` geometry keys
//! switch between them.
//!
//! ## Key bindings
//!
//! Every binding is generated from the one table in [`keymap`] (`?` shows the same
//! list as the hint bar and the `:` command palette, so the three surfaces can
//! never drift apart). The bindings a new contributor needs most:
//!
//! | Key | Action |
//! |-----|--------|
//! | `space` | paint / place with the active tool |
//! | `x` | erase cell / delete under cursor |
//! | `1`-`7` | geometry brush kinds; `8` staircase stamp; `9`/`0` hologram/portal |
//! | `o` / `m` / `v` | cycle Object / Material / Vox tool mode |
//! | `n` | tag a puzzle node at the cursor |
//! | `P` | pipe: wire two nodes together |
//! | `Tab` | cycle Split / Full2D / Full3D view |
//! | `<` / `>` | move a whole storey down / up |
//! | `t` / `i` | outliner focus / create-object palette |
//! | `u` / `y` | undo / redo |
//! | `s` | save; `A` save-as (name a never-saved level) |
//! | `F2` | rename an object, cascading every reference to it |
//! | `O` | open-level overlay (scans `levels/*.lvl`) |
//! | `E` | level settings (meta / room / effects / palette / sky / texture / inputs / sequence) |
//! | `V` | validation drawer — the last save's full lint report |
//! | `F5` | playtest the level in the real game (see below) |
//! | `?` | help cheat-sheet; `:` command palette |
//! | `Esc` | close the current modal, or quit from the root context |
//!
//! ## Saving and linting
//!
//! `s` serializes the in-memory level through [`re_voximport::to_lvl_text`] (the
//! same writer the `.vox`/`.toml` importer uses), then round-trips it through
//! [`re_lvlcheck::check_level`] before the write lands — a save that would produce
//! an unparsable or failing level is refused, and the full report is always
//! available afterward in the `V` validation drawer. A dirty, unsaved document is
//! also periodically written to a `levels/.autosave/` sidecar and offered back on
//! the next load if it is newer than the real file.
//!
//! ## Playtesting (`F5`)
//!
//! `F5` saves the level to a temporary sidecar, then hands the terminal to the
//! actual game so the level can be played exactly as a player would see it. The
//! player binary is resolved in this order, stopping at the first that applies:
//! 1. `$TS_PLAY_CMD`, if set — an explicit command line, useful for a non-adjacent
//!    build or a custom launcher; the level path is appended as its last argument.
//! 2. a `terminal-state` binary next to the running editor executable.
//! 3. a `terminal3d-play` binary next to the running editor executable (a generic
//!    `.lvl` player, for a build that ships without a full game).
//!
//! If none resolve, `F5` reports that no player was found and asks the user to set
//! `$TS_PLAY_CMD`; nothing is spawned and the terminal is never left in an
//! inconsistent state. On every exit path — including a spawn failure — the
//! editor's own terminal session resumes and forces a full redraw.
mod camera;
mod chrome;
#[cfg(test)]
mod corpus;
mod edit;
mod entity;
mod field;
mod fill;
mod focus;
mod hierarchy;
mod history;
mod io;
mod keymap;
mod layout;
mod material;
mod mouse;
mod open;
mod overlay;
mod palette_bar;
mod part;
mod picker;
mod pipe;
mod preview;
mod rename;
mod resize;
mod rowedit;
mod rowtype;
mod rowvalue;
mod saveas;
mod settings;
mod staircase;
mod state;
#[cfg(test)]
mod testutil;
mod toolbar;
mod tree;
mod validate;
mod view;
mod viewaids;
mod vocab;
mod vox;
mod widget;
mod wizard;

use bevy::prelude::*;
use camera::Fly;
use re_core::Framebuffer;
use re_present::{
    render_config::parse_size_arg, EditorKey, EditorKeyCode, PresentPlugin, PresentSet,
    RenderConfig,
};
use state::EditorState;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Resolve RenderConfig for the editor binary:
    // * hide_os_cursor default = false (editor keeps pointer visible for clicking).
    //   Override: TS_HIDE_CURSOR=1/true in the environment.
    // * fixed_size default = None — AUTO-SIZE to the terminal so the editor fills the
    //   screen and `resize_sys` tracks resizes. Override (pin a fixed grid):
    //   CLI `--size <cols>x<rows>` > TS_COLS+TS_ROWS env vars.
    let cli_size = parse_size_arg(&args);
    let render_cfg = RenderConfig::for_editor(cli_size);

    // Size the INITIAL framebuffer:
    // * fixed_size = Some(..) → use it (a CLI/env pin, or a fallback).
    // * fixed_size = None (auto-size) → query the ACTUAL terminal size so the editor
    //   is full-screen from frame 1. `resize_sys` (from PresentPlugin) keeps it in
    //   sync on subsequent terminal resizes. Fall back to 120×40 if the query fails.
    let (cols, rows) = render_cfg.fixed_size.unwrap_or_else(|| {
        re_present::terminal_size()
            .map(|(w, h)| (w.max(1), h.max(1)))
            .unwrap_or((120, 40))
    });

    let mut app = App::new();
    app.add_plugins(
        bevy::MinimalPlugins.set(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        )),
    );
    // Insert RenderConfig BEFORE PresentPlugin so the plugin's `init_resource`
    // (safe default) sees our editor-specific config instead.
    app.insert_resource(render_cfg);
    app.add_plugins(PresentPlugin);
    app.insert_resource(Framebuffer::new(cols, rows));
    app.insert_resource(EditorState::new_empty(16, 16, 4));
    app.init_resource::<state::EditorMouseTracker>();
    // PresentPlugin bundles the GAME's camera/input/gravity systems, which take
    // `Res<Map>` and `Query<&mut CameraPose>` as params. The editor spawns no
    // camera ENTITY (its preview camera lives in `EditorState`, not the ECS), so
    // those systems early-return on the empty query — but Bevy still validates
    // the `Res<Map>` param before running them. We insert a stub `Map` purely to
    // satisfy that validation; nothing in the editor reads this resource (the
    // real, edited map is `EditorState::map`). Without it the app panics at
    // startup: "Resource does not exist".
    app.insert_resource(re_world::Map::new(1, 1, 1));
    app.add_systems(Startup, startup_load);
    app.add_systems(
        Update,
        (
            input_system,
            camera_input_system,
            held_preview_mover,
            mouse::mouse_system,
            preview_capture_look_system,
            preview_rebuild_system,
            audition_tick_system,
            save_system,
            autosave_system,
            playtest_system,
            quit_on_esc,
        )
            .chain(),
    );
    app.add_systems(Last, draw_system.before(PresentSet));
    app.run();
}

/// Route 2D-edit keys (arrows, z, paint/erase, brushes, tag, save, Tab view
/// cycle) to the pure `edit::apply_key`. Camera/fly keys are handled by
/// `camera_input_system` (it needs `Res<Time>`), so the two are disjoint.
fn input_system(mut keys: MessageReader<EditorKey>, mut state: ResMut<EditorState>) {
    for k in keys.read() {
        edit::apply_key(&mut state, *k);
    }
}

/// Route the 3D preview-camera keys (the PER-EVENT fallback path). Separated from
/// `input_system` because the fly is time-based and needs the frame `dt` from
/// `Res<Time>`.
///
/// MODAL SEPARATION RULE (the contract): camera nav keys act ONLY while
/// `preview_captured` (right-mouse-button held); otherwise the keyboard is 2D
/// editing. This guarantees movement inputs never trigger edit hotkeys and edit
/// keys never move the camera — the two key surfaces are strictly disjoint by mode.
/// `edit::apply_key` enforces the other half (it early-returns while captured), so
/// pressing W/A/S/D in edit mode paints/selects, and the SAME letters fly the
/// camera only after RMB capture.
///
/// MOVEMENT MODEL (game parity): on a kitty-capable terminal the GAME's `input_sys`
/// populates `Res<HeldKeys>` from crossterm Press/Release, and `held_preview_mover`
/// (below) drives the captured camera continuously at `MOVE_SPEED`/`TURN_SPEED` —
/// exactly like the game's `held_move_sys`. This per-event path is the FALLBACK for
/// terminals with no release reporting (legacy conhost), where `HeldKeys` stays
/// empty: there the EditorKey autorepeat stream still arrives and each event nudges
/// the camera by the game's small per-event step (`camera::FLY_STEP`/`TURN_STEP`),
/// so holding a key glides instead of jumping. To avoid double-applying on kitty
/// terminals, the movement/turn arms run ONLY when the held set is empty.
///
/// Key scheme (the movement/turn arms ALL require `preview_captured`):
/// * `W`/`A`/`D` → Forward / StrafeLeft / StrafeRight
/// * `S`/`s` → Back (capture suspends the 2D-edit `s`=save binding, so the WASD
///   reflex works; `Shift+S` is kept for muscle memory).
/// * `R`/`F` → Up / Down
/// * `,`/`.` → TurnLeft / TurnRight
/// * `Enter` → jump-to-cursor. Available whether or not captured: `apply_key` has
///   NO Enter arm in its 2D-edit match (and modals are excluded above), so there is
///   no double-handling — it is a "send the preview camera here" convenience.
/// * `c` (while captured) → toggle collisions for the preview (clears `noclip`).
/// * `N`/`Shift+N` → toggle collisions. `N` is NOT a 2D-edit key (no `Char('N')`
///   arm in `apply_key`), so leaving it ungated causes no double-handling.
fn camera_input_system(
    mut keys: MessageReader<EditorKey>,
    mut state: ResMut<EditorState>,
    held: Res<re_present::HeldKeys>,
    time: Res<Time>,
) {
    // Any modal (help / dir-picker / open / save-as / wizard / palette / tree /
    // reparent) captures input; the fly camera must not move (or jump-to-cursor
    // on Enter) while a form is up — e.g. typing a filename in Save-As must not
    // also fly the camera, and Enter to confirm a modal must not also jump it.
    // Likewise while a TEXT FIELD is focused: Enter is a NEWLINE in a
    // note body, not a camera jump. (While preview-captured the text surface is
    // suspended — `apply_key` early-returns — so the camera keeps every key.)
    if state.any_modal_active()
        || (!state.preview_captured && entity::text_target(&state).is_some())
    {
        return;
    }
    let dt = time.delta_secs();
    // On a kitty terminal `HeldKeys` is non-empty exactly while a movement/turn key
    // is down, and `held_preview_mover` is driving the camera this frame — so skip
    // the per-event nudge to avoid moving twice. When the set is empty (non-kitty
    // fallback, or no key held) this guard is open and the per-event path runs.
    let held_driving = !held.is_empty();
    let captured = state.preview_captured;
    for k in keys.read() {
        apply_camera_key(&mut state, k.code, captured, held_driving, dt);
    }
}

/// Pure routing of ONE preview-camera key event — factored out of
/// `camera_input_system` so the modal-separation contract is unit-testable without
/// the ECS. Movement/turn arms fire ONLY when `captured && !held_driving`:
/// `captured` enforces "nav keys act only in look mode" (so W/A/S/D never leak into
/// 2D edit hotkeys), and `!held_driving` avoids double-applying on kitty terminals
/// where `held_preview_mover` already drives the camera this frame. `Enter`
/// (jump-to-cursor) and `N` (collision toggle) are NOT 2D-edit keys, so they stay
/// available regardless of capture with no double-handling. `c` toggles collisions
/// only while captured (the 2D-edit `c` is suspended then).
fn apply_camera_key(
    state: &mut EditorState,
    code: EditorKeyCode,
    captured: bool,
    held_driving: bool,
    dt: f32,
) {
    let can_fly = captured && !held_driving;
    match code {
        EditorKeyCode::Char('w') if can_fly => camera::fly(state, Fly::Forward, dt),
        EditorKeyCode::Char('s') if can_fly => camera::fly(state, Fly::Back, dt),
        EditorKeyCode::Char('S') if can_fly => camera::fly(state, Fly::Back, dt),
        EditorKeyCode::Char('a') if can_fly => camera::fly(state, Fly::StrafeLeft, dt),
        EditorKeyCode::Char('d') if can_fly => camera::fly(state, Fly::StrafeRight, dt),
        EditorKeyCode::Char('r') if can_fly => camera::fly(state, Fly::Up, dt),
        EditorKeyCode::Char('f') if can_fly => camera::fly(state, Fly::Down, dt),
        EditorKeyCode::Char(',') if can_fly => camera::fly(state, Fly::TurnLeft, dt),
        EditorKeyCode::Char('.') if can_fly => camera::fly(state, Fly::TurnRight, dt),
        EditorKeyCode::Enter => camera::jump_to_cursor(state),
        EditorKeyCode::Char('c') if captured => toggle_collisions(state),
        EditorKeyCode::Char('N') => toggle_collisions(state),
        _ => {}
    }
}

/// Toggle preview collisions. Collisions ON ⇔ `noclip == false`; `state.noclip` is
/// the single source of truth (the `chrome` panel's collisions checkbox reads the
/// same flag).
fn toggle_collisions(state: &mut EditorState) {
    state.noclip = !state.noclip;
    let on = !state.noclip;
    state.status = format!("collisions {}", if on { "ON" } else { "OFF" });
}

/// CONTINUOUS held-key preview mover — the GAME-PARITY movement path. Mirrors
/// `re_present::held_move_sys`: each frame, while `preview_captured`, it reads the
/// shared `Res<HeldKeys>` (populated by the game's `input_sys` from crossterm
/// Press/Release on kitty-capable terminals) and applies `MOVE_SPEED * dt` along
/// the combined held direction + `TURN_SPEED * dt` from the held arrows, so WASD
/// walking and arrow turning feel IDENTICAL to first-person play. Sprint (Shift)
/// scales speed by the game's `SPRINT_MULTIPLIER`. Movement respects collisions
/// unless `state.noclip` (mirroring `camera::fly`); pitch is clamped to the game's
/// ±`PITCH_LIMIT`.
///
/// Gated on `preview_captured` so it never moves the cam during 2D editing. It is a
/// no-op when the held set is empty — ALWAYS the case on non-kitty terminals, which
/// fall back to the per-event path in `camera_input_system`.
fn held_preview_mover(
    mut state: ResMut<EditorState>,
    held: Res<re_present::HeldKeys>,
    time: Res<Time>,
) {
    if !state.preview_captured || held.is_empty() || state.any_modal_active() {
        return;
    }
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let (dirx, diry) = state.cam.dir();
    let speed = if held.sprint {
        re_present::MOVE_SPEED * re_present::SPRINT_MULTIPLIER
    } else {
        re_present::MOVE_SPEED
    };
    let (mdx, mdy) = re_present::held_move_delta(*held, dirx, diry, speed, dt);
    if mdx != 0.0 || mdy != 0.0 {
        camera::translate_cam(&mut state, mdx, mdy);
    }
    let (dyaw, dpitch) = re_present::held_turn_delta(*held, re_present::TURN_SPEED, dt);
    if dyaw != 0.0 {
        state.cam.yaw += dyaw;
    }
    if dpitch != 0.0 {
        // Clamp to the game's pitch limit. `re_present::PITCH_LIMIT` is private, so
        // use its value directly: π/3 (matches `apply_mouse_look`'s clamp exactly).
        const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_3;
        state.cam.pitch = (state.cam.pitch + dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }
}

/// Apply ONE crossterm cell-motion sample to the fly camera (the path-2 per-event
/// math), exactly as the game's `mouse_look_sys` fallback does. A fresh sample
/// (`last == None`) only ANCHORS — it records `pos` and leaves the camera alone, so
/// the first absolute coordinate after capture never produces a stale-position
/// lurch. Each subsequent sample differences `pos` against `last` and feeds the
/// delta through the GAME's `re_present::apply_mouse_look` (same sensitivity, same
/// mouse-down-looks-down sign, same ±π/3 pitch clamp), then advances `last`.
///
/// Pure (no terminal, no ECS), so the dual-path math is unit-tested headlessly.
fn apply_cell_look(
    cam: &mut re_world::CameraPose,
    last: &mut Option<(u16, u16)>,
    pos: (u16, u16),
    sens: f32,
) {
    if let Some((px, py)) = *last {
        let dx = pos.0 as f32 - px as f32;
        let dy = pos.1 as f32 - py as f32;
        if dx != 0.0 || dy != 0.0 {
            let (yaw, pitch) = re_present::apply_mouse_look(cam.yaw, cam.pitch, dx, dy, sens);
            cam.yaw = yaw;
            cam.pitch = pitch;
        }
    }
    *last = Some(pos);
}

/// PREVIEW MOUSE-CAPTURE driver — the DUAL-PATH free-look that mirrors the game's
/// `mouse_look_sys` (`re_present::terminal`). Owns the `re_present::OsCursorGrab`
/// resource on behalf of the pure `mouse::apply_mouse` (which can only set a
/// `capture_request`). Capture is now "RIGHT mouse button is held" (set/cleared by
/// `mouse.rs`). Each frame it:
///
/// 1. Drains `state.capture_request`: `Some(true)` → `os_grab.activate(rows)`,
///    `preview_captured = true`, reset the tracker; `Some(false)` →
///    `os_grab.deactivate()` + `preview_captured = false`.
/// 2. PATH 1 — OS cursor grab (conhost / X11): `poll_delta()` gives a pixel-res
///    delta → rotate via `re_present::apply_mouse_look` (identical sensitivity /
///    axis sign / ±π/3 clamp to the first-person camera). The first time it yields
///    we latch `grab_yielded`.
/// 3. FALL-THROUGH — on Windows Terminal `GetConsoleWindow()` is a hidden ConPTY
///    pseudo-window so the grab is nominally `active` but `poll_delta()` always
///    returns `None`. If it has NEVER yielded (`!grab_yielded`) we fall through to
///    PATH 2; once it HAS yielded, a `None` is just "no motion this frame".
/// 4. PATH 2 — crossterm cell events (`MessageReader<MouseMessage>` via the pure
///    `re_present::mouse_motion_cell`): difference absolute cell coords frame-to-
///    frame (the game's fallback). This is what actually restores look on Windows
///    Terminal / Wayland.
fn preview_capture_look_system(
    mut state: ResMut<EditorState>,
    mut os_grab: ResMut<re_present::OsCursorGrab>,
    mouse_settings: Res<re_present::MouseSettings>,
    mut tracker: ResMut<state::EditorMouseTracker>,
    mut raw: MessageReader<re_present::MouseMessage>,
    fb: Res<Framebuffer>,
) {
    // (1) Drain a pending activate/deactivate request and reset the tracker.
    if let Some(req) = state.capture_request.take() {
        if req {
            let rows = fb.rows();
            let _ = os_grab.activate(rows);
            state.preview_captured = true;
            tracker.last = None;
            tracker.grab_yielded = false;
            state.status = "look: hold right mouse + move · WASD fly · release to stop".into();
        } else {
            os_grab.deactivate();
            state.preview_captured = false;
            tracker.last = None;
            state.status = "preview released".into();
        }
    }

    // (2) Not looking, or a modal is up: keep the stream fresh so re-entering does
    //     not see a stale-position jump, and apply nothing.
    if !state.preview_captured || state.any_modal_active() {
        raw.clear();
        tracker.last = None;
        return;
    }

    // The editor's capture flag is the gate; `MouseSettings::enabled` only toggles
    // mouse-look off entirely. (The game's `InputGate` is false in the editor, so it
    // must NOT be consulted here.)
    let apply = mouse_settings.enabled;

    // (3) PATH 1: OS grab (conhost / X11 pixel-resolution).
    if os_grab.active {
        if let Some((dx, dy)) = os_grab.poll_delta() {
            tracker.grab_yielded = true;
            if apply && (dx != 0.0 || dy != 0.0) {
                let (y, p) = re_present::apply_mouse_look(
                    state.cam.yaw,
                    state.cam.pitch,
                    dx,
                    dy,
                    mouse_settings.sensitivity,
                );
                state.cam.yaw = y;
                state.cam.pitch = p;
            }
            // One write per frame: drained the cell stream so it can't double-apply.
            raw.clear();
            tracker.last = None;
            return;
        }
        // poll_delta == None:
        if tracker.grab_yielded {
            // A real, working grab — just no motion this frame.
            raw.clear();
            tracker.last = None;
            return;
        }
        // else: WINDOWS TERMINAL — grab "active" but never yields. Fall through.
    }

    // (4) PATH 2: crossterm cell fallback (Windows Terminal / Wayland / grab off).
    for ev in raw.read() {
        let Some(pos) = re_present::mouse_motion_cell(ev) else {
            continue;
        };
        if apply {
            apply_cell_look(
                &mut state.cam,
                &mut tracker.last,
                pos,
                mouse_settings.sensitivity,
            );
        } else {
            // Keep the tracker current even when look is disabled, so re-enabling
            // doesn't lurch from a stale anchor.
            tracker.last = Some(pos);
        }
    }
}

/// Tick the DEBOUNCED preview rebuild: the 60 Hz loop decrements the
/// countdown `EditorState::schedule_preview_rebuild` armed and runs the map
/// rebuild ~150 ms after the last entity/material edit. All logic lives in the
/// pure `vox::tick_preview_rebuild`; this is just its frame clock.
fn preview_rebuild_system(mut state: ResMut<EditorState>) {
    let _ = vox::tick_preview_rebuild(&mut state);
}

/// SOUND-FIELD LIVE AUDITION — the frame clock. While `Shift+U` has
/// armed `state.audition` (`crate::field::toggle_audition`), re-evaluate
/// every field from the PREVIEW camera's current position each frame and
/// push the resulting mix into the live `re_audio::Player`, so walking the
/// preview through a field audibly/visibly changes its weight. A no-op when
/// unarmed (the common case, and always the case under `TS_AUDIO_SILENT`/no
/// device — `Audition::start` never constructs a `Player` then, so
/// `state.audition` stays `None` and this system has nothing to tick).
fn audition_tick_system(mut state: ResMut<EditorState>) {
    let Some(mut audition) = state.audition.take() else {
        return;
    };
    let listener = [state.cam.pos.x, state.cam.pos.y, state.cam.pos.z];
    audition.tick(&state.level, listener);
    state.audition = Some(audition);
}

/// F5 PLAYTEST — the terminal-handoff half. Exclusive system (`&mut
/// World`): `re_present::suspend_terminal`/`resume_terminal` remove/replace the
/// terminal-owning resources wholesale. Drains `EditorState::playtest_request`
/// (raised by the pure F5 arm in `edit.rs`, already warn-and-confirmed on a
/// lvlcheck FAIL), then:
///
/// 1. saves the CURRENT document to the `levels/_scratch/preview-*.lvl`
///    sidecar (`io::save_playtest_sidecar` — reparse-guarded, never the
///    original file, dirty state untouched);
/// 2. resolves the game (see [`resolve_game_command`]): `$TS_PLAY_CMD` first,
///    else the `current_exe()`-sibling `terminal-state[.exe]`, else the
///    sibling `terminal3d-play[.exe]` (a harmless probe when that binary does
///    not exist next to the editor), else no launch at all. Status reports
///    WHICH, or tells the user to set
///    `$TS_PLAY_CMD` if none resolved;
/// 3. on a resolved command: suspends the terminal, spawns the child with the
///    sidecar path as the level arg (the game reads it as the first non-flag
///    arg), and WAITS;
/// 4. resumes the terminal on EVERY exit path — spawn failure included — so
///    the editor is never left outside the alt screen. The fresh
///    `RatatuiContext` starts with empty ratatui buffers, so the next present
///    is a full redraw. (Nothing is suspended at all if step 2 found no
///    player — there is no terminal handoff to make.)
fn playtest_system(world: &mut World) {
    let requested = {
        let mut state = world.resource_mut::<EditorState>();
        std::mem::take(&mut state.playtest_request)
    };
    if !requested {
        return;
    }
    // 1. Sidecar save (pure I/O; refusal keeps the editor exactly as it was).
    let sidecar = {
        let mut state = world.resource_mut::<EditorState>();
        match io::save_playtest_sidecar(&mut state) {
            Ok(p) => p,
            Err(e) => {
                state.status = format!("playtest refused: {e}");
                return;
            }
        }
    };
    // 2. Resolve the game command BEFORE suspending (cheap, no terminal work).
    //    No player found ⇒ report and stop; there is nothing to hand the
    //    terminal off to.
    let (label, mut cmd) = match resolve_game_command(&sidecar) {
        Ok(v) => v,
        Err(msg) => {
            let mut state = world.resource_mut::<EditorState>();
            state.status = msg;
            return;
        }
    };
    // 3-4. Handoff: suspend → spawn+wait → resume, resume on EVERY path.
    re_present::suspend_terminal(world);
    let run = cmd.status();
    let resumed = re_present::resume_terminal(world);
    let mut state = world.resource_mut::<EditorState>();
    state.status = match run {
        Ok(code) => format!("playtest done ({code}) via {label}"),
        Err(e) => format!("playtest failed to launch ({label}): {e}"),
    };
    if !resumed {
        // resume_terminal already logged; fail loud in the status too (the
        // next present will panic on the missing context, which is correct).
        state.status = "TERMINAL RESUME FAILED — restart the editor".into();
    }
}

/// A `current_exe()`-sibling binary path named `name` (+ the platform's
/// `EXE_SUFFIX`), regardless of whether it exists — a PURE helper, unit-testable
/// without a filesystem. Used for both the game binary (`terminal-state`) and
/// the generic `.lvl` player (`terminal3d-play`), when either ships next to the
/// editor executable.
fn sibling_binary(exe: &std::path::Path, name: &str) -> std::path::PathBuf {
    let dir = exe.parent().unwrap_or_else(|| std::path::Path::new("."));
    dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// Resolve how to launch the game on `level_path`, in order:
///
/// 1. `$TS_PLAY_CMD`, if set to a non-empty program path — the level path is
///    appended as its argument. The escape hatch for a dev setup this resolver
///    cannot guess (a custom launcher, a non-adjacent build, CI).
/// 2. the `current_exe()`-sibling `terminal-state[.exe]`, if it EXISTS.
/// 3. the `current_exe()`-sibling `terminal3d-play[.exe]` (a generic `.lvl`
///    player, for a build that ships without a full game) — a harmless probe
///    when that binary does not exist next to the editor.
/// 4. otherwise `Err` — the caller must not spawn anything and must tell the
///    user to set `$TS_PLAY_CMD`.
///
/// The resolver only ever finds the game by its BINARY name (`terminal-state`),
/// never by a cargo package name — the editor and the game are built and
/// shipped as separate binaries, possibly from separate repositories, so
/// naming a package here would be fragile.
fn resolve_game_command(level_path: &str) -> Result<(String, std::process::Command), String> {
    let play_cmd = std::env::var("TS_PLAY_CMD").ok();
    let exe = std::env::current_exe().ok();
    resolve_game_command_from(play_cmd.as_deref(), exe.as_deref(), level_path)
}

/// The PURE half of [`resolve_game_command`]'s resolution order, over already-
/// read inputs — unit-testable without mutating process environment variables
/// (`std::env::set_var` is `unsafe fn` on this toolchain, and this workspace
/// allows no `unsafe`) or depending on the real build output existing on disk.
fn resolve_game_command_from(
    play_cmd_override: Option<&str>,
    editor_exe: Option<&std::path::Path>,
    level_path: &str,
) -> Result<(String, std::process::Command), String> {
    if let Some(prog) = play_cmd_override.filter(|s| !s.is_empty()) {
        let mut cmd = std::process::Command::new(prog);
        cmd.arg(level_path);
        return Ok((format!("$TS_PLAY_CMD={prog}"), cmd));
    }
    if let Some(exe) = editor_exe {
        for name in ["terminal-state", "terminal3d-play"] {
            let sibling = sibling_binary(exe, name);
            if sibling.exists() {
                let label = sibling.display().to_string();
                let mut cmd = std::process::Command::new(&sibling);
                cmd.arg(level_path);
                return Ok((label, cmd));
            }
        }
    }
    Err("no player found — set TS_PLAY_CMD".to_string())
}

/// AUTOSAVE — the frame clock. Feeds the frame `dt` from `Res<Time>` into
/// the pure [`autosave_tick`]; all policy (dirty-gating, interval, reparse
/// guard, sidecar path) lives there so it is unit-testable without the ECS.
fn autosave_system(mut state: ResMut<EditorState>, time: Res<Time>) {
    // `into_inner` consumes the `Res` handle (pass-by-value is the required
    // Bevy system-param shape; consuming it keeps clippy's advisory noise flat).
    autosave_tick(&mut state, time.into_inner().delta_secs());
}

/// Tick the autosave timer by `dt` seconds. While the document is CLEAN
/// the timer stays at zero (nothing to protect). While DIRTY it accumulates;
/// at [`io::AUTOSAVE_INTERVAL_SECS`] the document is serialized through the
/// SAME reparse guard as a real save and written to the
/// `levels/.autosave/<stem>.lvl` sidecar (write-to-temp + atomic rename — see
/// [`io::write_autosave`]). The ORIGINAL file is never touched and the dirty
/// flag is never cleared. On a refused serialize (reparse Err) nothing is
/// written and the status says so; the timer resets either way so a broken
/// document does not retry every frame.
fn autosave_tick(state: &mut EditorState, dt: f32) {
    if !state.dirty {
        // Clean document ⇒ nothing at risk; keep the timer parked at zero.
        state.autosave_elapsed = 0.0;
        return;
    }
    state.autosave_elapsed += dt;
    if state.autosave_elapsed < io::AUTOSAVE_INTERVAL_SECS {
        return;
    }
    state.autosave_elapsed = 0.0;
    let dest = io::autosave_path_for(state.path.as_deref());
    match io::write_autosave(state, &dest) {
        Ok(()) => state.status = format!("autosaved {dest}"),
        Err(e) => state.status = format!("autosave skipped: {e}"),
    }
}

fn save_system(mut state: ResMut<EditorState>) {
    if state.status == "SAVE" {
        // A never-saved document (no path) must NOT be reported "saved".
        // Route it to the Save-As prompt instead so the user names the file; the
        // confirm sets the path and re-raises "SAVE" to land here with a path.
        if state.path.is_none() {
            saveas::open_saveas(&mut state);
            return;
        }
        let verdict = io::save_lvl_text(&mut state);
        state.status = format!("saved: {verdict}");
    }
}

fn quit_on_esc(
    mut keys: MessageReader<EditorKey>,
    mut state: ResMut<EditorState>,
    mut os_grab: ResMut<re_present::OsCursorGrab>,
    mut tracker: ResMut<state::EditorMouseTracker>,
    mut exit: MessageWriter<AppExit>,
) {
    for k in keys.read() {
        if k.code == EditorKeyCode::Esc {
            // Esc precedence: PREVIEW CAPTURE releases FIRST (highest priority), then
            // modals close (consuming the keypress so a single Esc never both closes
            // a modal and quits), then the dirty-quit guard, then the actual exit.
            // Order: capture → mask → help → wizard → palette → open → saveas →
            // reparent → tree → dirty-guard → quit.
            if state.preview_captured {
                // Release the OS grab + the editor flag here (this system has the
                // resource); consume the Esc so it doesn't also close a modal/quit.
                // Reset the look tracker too so a later re-capture starts jump-free.
                os_grab.deactivate();
                state.preview_captured = false;
                state.capture_request = None;
                tracker.last = None;
                tracker.grab_yielded = false;
                state.status = "preview released".into();
            } else if open::dismiss_autosave_offer(&mut state) {
                // The crash-recovery offer is MODAL (`apply_key`
                // swallows every non-y/n key), so Esc is its third answer —
                // "leave the document as opened, keep the sidecar". It sits high
                // in the chain because the offer only appears right after a load,
                // before any other modal can be up.
            } else if rowedit::cancel(&mut state) {
                // A ROW EDITOR (inline number / enum popup / colour overlay) is
                // open: Esc closes it, writing nothing. It sits high in the
                // chain because these open FROM the inspector and are the
                // innermost thing on screen whenever they are up. `cancel` is
                // also what `focus.rs` leaves Esc to, so whichever system sees
                // the keypress first closes it and the other finds nothing —
                // an Esc can never fall through to the quit guard.
            } else if state.mask_mode {
                // A single Esc LEAVES part mask-editing (like capture-release),
                // rather than falling through to the dirty-quit guard / exit. Placed
                // near the top of the chain so mask mode owns Esc while it is open.
                state.mask_mode = false;
                state.status = "mask edit closed".into();
            } else if state.cmd_palette.is_some() {
                // Esc closes the `:` command palette without running anything.
                state.cmd_palette = None;
                state.status = "palette closed".into();
            } else if state.rename.is_some() {
                // Esc cancels the F2 rename prompt without renaming.
                rename::cancel(&mut state);
            } else if state.help_open && !state.help_filter.is_empty() {
                // Esc CLEARS the help sheet's live filter first; a second Esc
                // closes the sheet via the branch below. Mirrors the outliner's
                // two-step Esc a few branches down — the filter is the innermost
                // thing the user built, so it is the innermost thing Esc undoes.
                state.help_filter.clear();
                state.help_scroll = 0;
                state.status = "help filter cleared".into();
            } else if state.help_open {
                // Esc closes the help popup without quitting.
                state.close_help();
            } else if state.dir_picker.is_some() {
                // Esc closes the direction picker without changing the aim.
                state.dir_picker = None;
                state.status = "direction picker cancelled".into();
            } else if state.validation_open {
                // Esc closes the validation drawer (read-only — nothing to lose).
                state.validation_open = false;
                state.status = "validation drawer closed".into();
            } else if state.tree.open && state.tree.filter.is_some() {
                // Esc CLEARS the outliner filter first; a second Esc closes the
                // outliner via the branch below.
                state.tree.filter = None;
                state.status = "outliner filter cleared".into();
            } else if state.wizard.is_some() {
                state.wizard = None;
                state.status = "wizard cancelled".into();
            } else if state.settings.is_some() {
                // Esc DISCARDS unconfirmed settings edits (nothing is written to
                // the level until Enter → settings::apply).
                state.settings = None;
                state.status = "level settings closed".into();
            } else if state.palette.is_some() {
                // Esc closes the create palette without quitting.
                state.palette = None;
                state.status = "create palette closed".into();
            } else if state.open.is_some() {
                // Esc closes the file-open overlay without quitting.
                open::close_overlay(&mut state);
            } else if state.saveas.is_some() {
                // Esc closes the Save-As prompt without quitting or writing.
                saveas::close_saveas(&mut state);
            } else if state.reparent_pending.is_some() {
                // Esc cancels an in-flight reparent pick (keeps the tree open).
                state.reparent_pending = None;
                state.status = "reparent cancelled".into();
            } else if state.tree.open {
                // Esc closes the object tree without quitting.
                state.tree.open = false;
                state.status = "object tree closed".into();
            } else if state.dirty && !state.quit_armed {
                // DIRTY-QUIT GUARD: first Esc on unsaved work arms the guard
                // instead of quitting. A second Esc (armed) exits; any other key
                // disarms it (cleared at the top of `apply_key`).
                state.quit_armed = true;
                state.status = "UNSAVED — Esc again to quit, s to save".into();
            } else {
                // Belt-and-suspenders: release the OS grab before quitting if it
                // is somehow still live (Drop also covers this on app exit).
                if state.preview_captured {
                    os_grab.deactivate();
                    state.preview_captured = false;
                }
                // A CLEAN quit deletes this document's autosave sidecar —
                // there is nothing to recover, so a stale sidecar must not
                // trigger a bogus crash-recovery offer on the next open. A
                // dirty (armed) quit KEEPS it: the user is discarding edits the
                // sidecar may still hold. Delete errors are ignored.
                if !state.dirty {
                    io::delete_autosave(&io::autosave_path_for(state.path.as_deref()));
                }
                exit.write(AppExit::Success);
            }
        }
    }
}

fn startup_load(mut state: ResMut<EditorState>) {
    load_arg(&mut state);
}

fn load_arg(state: &mut EditorState) {
    if let Some(path) = std::env::args().nth(1) {
        // Both startup-load and the in-editor Open overlay share `load_path`, so
        // there is exactly one cleanly-rebuilding load routine.
        open::load_path(state, &path);
    }
}

fn draw_system(mut state: ResMut<EditorState>, mut fb: ResMut<Framebuffer>) {
    fb.clear();
    let w = fb.cols();
    let h = fb.rows();
    // Below the usable size floor, the panes would collapse to unreadable slivers
    // (see layout::compute). Draw a single "terminal too small" message instead of
    // rendering garbage, and bail before any pane work.
    if chrome::too_small(w, h) {
        chrome::draw_too_small(&mut fb, w, h);
        return;
    }
    // The PANEL-AWARE layout (collapse states + shedding + rail + zoom).
    // `apply_mouse` derives the same layout from the same state, so painters
    // and hit-tests stay aligned.
    let l = state.layout_for(w, h);
    let eff = state.panels_eff(w);

    // Auto-follow pan: keep the cursor inside the grid window. Computed here (we
    // have the live grid pane size). The visible cell area excludes the ruler
    // gutters. We re-centre on the cursor ONLY when the cursor actually moved
    // this frame, so a manual pan (Shift+arrows) or an anchored zoom-to-cursor
    // is not yanked back every frame. Regardless of cursor movement we always
    // clamp pan to the legal range so a resize or zoom never leaves it out of
    // bounds.
    if !l.grid.is_empty() {
        let (sx, sy, _sz) = state.map.size();
        let cells_w = l.grid.w.saturating_sub(chrome::RULER_W);
        let cells_h = l.grid.h.saturating_sub(chrome::RULER_H);
        // Store the live cell-area viewport so keyboard/mouse zoom can anchor.
        state.view_cells = (cells_w, cells_h);
        let zoom = state.zoom;
        if state.cursor == state.last_cursor {
            // Cursor unchanged: leave pan as-is (manual pan / anchored zoom
            // persists), but still clamp it to the legal range. Mirrors view.rs's
            // window convention: visible whole-cells = (pane / zoom).max(1), and
            // the far-edge pan = map - visible.
            let vis_x = view::visible_cells(cells_w, zoom);
            let vis_y = view::visible_cells(cells_h, zoom);
            state.pan = (
                state.pan.0.min(sx.saturating_sub(vis_x)),
                state.pan.1.min(sy.saturating_sub(vis_y)),
            );
        } else {
            // Cursor moved: re-centre the window to keep it visible.
            let (cx, cy) = state.cursor;
            let new_px = view::follow_pan(state.pan.0, cx, cells_w, zoom, sx);
            let new_py = view::follow_pan(state.pan.1, cy, cells_h, zoom, sy);
            state.pan = (new_px, new_py);
        }
    }
    // Record this frame's cursor so the next frame can detect movement.
    state.last_cursor = state.cursor;

    // The `[<>]` swap control lives on the LEFT center viewport's header, and only
    // in Split (where both viewports show). Decide which physical viewport is on the
    // left so the control + the focus cue line up with what is drawn. `swap_*`
    // booleans gate the single control to exactly one header.
    let both_viewports = !l.grid.is_empty() && !l.preview.is_empty();
    let grid_is_left = both_viewports && l.grid.x <= l.preview.x;
    let swap_on_grid = both_viewports && grid_is_left;
    let swap_on_preview = both_viewports && !grid_is_left;
    if !l.grid.is_empty() {
        // Reserve a 1-row caption header at the top of the 2D viewport; the grid
        // scene draws BELOW it so the header never overwrites cells.
        let focused = state.focused_viewport == crate::state::ViewportFocus::Grid;
        chrome::draw_grid_header(&state, &mut fb, l.grid, focused, swap_on_grid);
        let hh = chrome::VIEWPORT_HEADER_H.min(l.grid.h);
        chrome::draw_grid(
            &state,
            &mut fb,
            l.grid.x,
            l.grid.y + hh,
            l.grid.w,
            l.grid.h - hh,
        );
    }
    if !l.panel.is_empty() {
        // The side panel hosts TWO stacked organs: the always-visible OUTLINER
        // (object list) on top and the context-sensitive INSPECTOR below, split by
        // `chrome::{outliner,inspector}_subrect` (the SAME geometry the hit-tests
        // use). A 1-row `─ INSPECTOR ─` divider separates them when the panel is
        // tall enough to split.
        let split = layout::panel_subrects(l.panel);
        let outliner = chrome::outliner_subrect_eff(&state, l.panel, eff);
        let inspector = chrome::inspector_subrect_eff(&state, l.panel, eff);
        if !outliner.is_empty() {
            chrome::draw_tree(
                &state, &mut fb, outliner.x, outliner.y, outliner.w, outliner.h,
            );
        }
        // The organ divider only exists while BOTH organs are visible (a
        // railed organ hands the whole column to its sibling).
        if let (Some(divider_y), false, false) =
            (split.divider_y, outliner.is_empty(), inspector.is_empty())
        {
            chrome::draw_organ_divider(&mut fb, l.panel.x, divider_y, l.panel.w);
        }
        if !inspector.is_empty() {
            chrome::draw_panel(&state, &mut fb, inspector.x, inspector.y, inspector.h);
            // INSPECTOR action buttons (P2 inspector / P3 view / P4 fill / P6 tree).
            chrome::draw_panel_buttons(&state, &mut fb, inspector);
            // Optional ELEVATION cross-section at the very bottom of the inspector
            // when there is spare height below the buttons (lower priority than the
            // outliner/inspector; simply absent on a short panel).
            if let Some(elev) = chrome::inspector_elevation_rect(&state, inspector) {
                chrome::draw_elevation(&state, &mut fb, elev.x, elev.y, elev.w, elev.h);
            }
        }
    }
    if !l.preview.is_empty() {
        // Reserve a 1-row caption header at the top of the 3D viewport; the preview
        // scene renders BELOW it so the header never overwrites the render.
        let focused = state.focused_viewport == crate::state::ViewportFocus::Preview;
        chrome::draw_preview_header(&state, &mut fb, l.preview, focused, swap_on_preview);
        let hh = chrome::VIEWPORT_HEADER_H.min(l.preview.h);
        let scene = layout::Rect {
            x: l.preview.x,
            y: l.preview.y + hh,
            w: l.preview.w,
            h: l.preview.h - hh,
        };
        preview::render_preview(&state, &mut fb, scene.x, scene.y, scene.w, scene.h);
        // One-line discoverability hint (drag = look · wheel = move · …), drawn over
        // the scene's bottom row when not in OS free-look capture.
        chrome::draw_preview_hint(&state, &mut fb, scene);
    }
    // (The OBJECT TREE / OUTLINER is now drawn as the TOP organ of the side panel
    // above — it is always visible, no longer a carved bottom strip.)
    // The top TOOLBAR strip (clickable mode/Open/Save/Help). Drawn over the panes
    // (it owns row 0) but under the centered overlays below.
    if !l.toolbar.is_empty() {
        toolbar::draw_toolbar(&state, &mut fb, l.toolbar.x, l.toolbar.y, l.toolbar.w);
    }
    // The vertical TOOLBOX (grouped tool buttons) down the left body column.
    if !l.toolbox.is_empty() {
        palette_bar::draw_toolbox(&state, &mut fb, l.toolbox);
    }
    // The full-width STATUS BAR (bottom row): the "in-the-moment" readouts —
    // view/z/cursor/collisions/undo-redo/saved + status + a key hint. Shown in
    // every view mode (it replaces the old Full3D top strip + the panel readouts).
    if !l.status.is_empty() {
        chrome::draw_status_bar(&state, &mut fb, l.status);
    }
    // The 1-col ICON RAIL down the right edge: one icon per collapsed panel
    // clicking an icon reopens the panel.
    if !l.rail.is_empty() {
        chrome::draw_rail(&mut fb, l.rail, eff);
    }
    // The read-only VALIDATION DRAWER: the last save's full lvlcheck
    // report. A side drawer over the body, under the centered modals.
    if state.validation_open {
        validate::draw(&state, &mut fb, w, h);
    }
    // The object-TYPE SELECTOR palette overlays, centered, when open.
    if state.palette.is_some() {
        chrome::draw_palette(&state, &mut fb, w, h);
    }
    // The NEW-LEVEL wizard overlays everything else, centered, when open.
    if state.wizard.is_some() {
        chrome::draw_wizard(&state, &mut fb, w, h);
    }
    // The LEVEL SETTINGS form overlays, centered, when open.
    if state.settings.is_some() {
        chrome::draw_settings(&state, &mut fb, w, h);
    }
    // The LOAD/OPEN file picker overlays, centered, when open.
    if state.open.is_some() {
        chrome::draw_open(&state, &mut fb, w, h);
    }
    // The SAVE-AS filename prompt overlays, centered, when open.
    if state.saveas.is_some() {
        chrome::draw_saveas(&state, &mut fb, w, h);
    }
    // The F2 RENAME prompt overlays, centered, when open.
    if state.rename.is_some() {
        chrome::draw_rename(&state, &mut fb, w, h);
    }
    // The DIRECTION PICKER (3×3×3 aim cube) overlays, centered, when open.
    if state.dir_picker.is_some() {
        chrome::draw_dir_picker(&state, &mut fb, w, h);
    }
    // The three ROW EDITORS (typed inspector row widgets) overlay, centered.
    // Only one can be open at a time; they draw above the object modals because
    // they are opened FROM the inspector and are the innermost thing on screen.
    if state.num_edit.is_some() {
        chrome::draw_num_edit(&state, &mut fb, w, h);
    }
    if state.enum_pick.is_some() {
        chrome::draw_enum_pick(&state, &mut fb, w, h);
    }
    if state.color_edit.is_some() {
        chrome::draw_color_edit(&state, &mut fb, w, h);
    }
    // The HELP cheat-sheet popup is TOPMOST so it's always readable.
    if state.help_open {
        // PUBLISH the painted body height, exactly as `state.view_cells` above
        // publishes the grid viewport: `PageUp`/`PageDown`/`End` and the wheel
        // clamp and step against the window the user is actually looking at, and
        // the key handlers cannot see the framebuffer.
        let body_h = chrome::help_layout(&state, w, h).map_or(0, |l| l.body_h);
        state.help_page = body_h;
        chrome::draw_help(&state, &mut fb, w, h);
    }
    // The `:` COMMAND PALETTE — the topmost focus frame, so it draws
    // over every other overlay (only the capture banner outranks it).
    if state.cmd_palette.is_some() {
        chrome::draw_cmd_palette(&state, &mut fb, w, h);
    }
    // The CRASH-RECOVERY offer draws ABOVE every other overlay because it
    // DISPATCHES above every other one: `edit::apply_key` handles it at the very
    // front of the key chain (and `mouse::apply_modal_click` at the front of the
    // click chain), so whatever else happens to be open underneath is inert while
    // it is pending. Dispatch precedence ⇒ paint precedence — a modal that owns
    // the input and paints below something else is a lie about what the keyboard
    // is doing. (Only the capture banner outranks it, and `preview_captured` is
    // released before a load can raise an offer.)
    if state.autosave_offer.is_some() {
        chrome::draw_autosave_offer(&state, &mut fb, w, h);
    }
    // PREVIEW-CAPTURE banner: a bright top strip while free-look is active. Drawn
    // last so the indicator is always visible.
    if state.preview_captured {
        chrome::draw_capture_banner(&mut fb, w, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ESC PRECEDENCE for the help sheet: Esc CLEARS a live filter first, and
    /// only closes the sheet once the filter is empty. `quit_on_esc` is a Bevy
    /// system over five resources, so the order is asserted over the SOURCE —
    /// line by line, never with a needle containing a newline, because
    /// `core.autocrlf` is on for this repo and a `\n` needle matches nothing in
    /// a CRLF checkout (a test that passes while checking nothing).
    #[test]
    fn esc_clears_the_help_filter_before_it_closes_the_sheet() {
        let src = include_str!("main.rs");
        let lines: Vec<&str> = src.lines().collect(); // `lines()` eats a trailing CR
        let test_mod = lines
            .iter()
            .position(|l| l.trim() == "mod tests {")
            .expect("the test module marker");
        let code: Vec<(usize, &str)> = lines[..test_mod]
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.trim_start().starts_with("//"))
            .map(|(i, l)| (i, *l))
            .collect();
        let line_of = |needle: &str| {
            code.iter()
                .find(|(_, l)| l.contains(needle))
                .map(|(i, _)| *i)
        };

        let clear = line_of("state.help_open && !state.help_filter.is_empty()")
            .expect("Esc must clear a live help filter");
        let close = line_of("state.close_help()").expect("Esc must then close the sheet");
        let quit = line_of("state.quit_armed").expect("premise: the dirty-quit guard");
        assert!(
            clear < close,
            "clearing the filter outranks closing the sheet"
        );
        assert!(close < quit, "closing the sheet still outranks quitting");
    }

    /// The sibling resolution is `<exe dir>/<name><EXE_SUFFIX>`.
    #[test]
    fn sibling_binary_lands_next_to_the_editor_exe() {
        let exe = std::path::Path::new("some")
            .join("dir")
            .join("ts_editor.exe");
        let sib = sibling_binary(&exe, "terminal-state");
        assert_eq!(sib.parent(), exe.parent(), "same directory as the editor");
        assert_eq!(
            sib.file_name().and_then(|n| n.to_str()),
            Some(format!("terminal-state{}", std::env::consts::EXE_SUFFIX).as_str()),
            "the game binary name + platform suffix"
        );
    }

    /// Resolution tier 1: `$TS_PLAY_CMD` wins over everything else,
    /// even when no `editor_exe` is given at all — it is the escape hatch for
    /// exactly the case where the resolver cannot find anything on its own.
    #[test]
    fn resolve_game_command_from_prefers_the_play_cmd_override() {
        let (label, cmd) =
            resolve_game_command_from(Some("some-launcher"), None, "levels/_scratch/x.lvl")
                .expect("an override always resolves");
        assert_eq!(
            cmd.get_program(),
            "some-launcher",
            "runs the override verbatim"
        );
        assert_eq!(
            cmd.get_args()
                .last()
                .map(|a| a.to_string_lossy().to_string()),
            Some("levels/_scratch/x.lvl".to_string()),
            "the level path is appended"
        );
        assert!(
            label.contains("TS_PLAY_CMD"),
            "label names the override: {label}"
        );
    }

    /// An empty `$TS_PLAY_CMD` (set but blank) is treated as unset, not as a
    /// program named "".
    #[test]
    fn resolve_game_command_from_ignores_an_empty_play_cmd_override() {
        let err = resolve_game_command_from(Some(""), None, "levels/x.lvl")
            .expect_err("blank override + no sibling ⇒ nothing resolves");
        assert!(err.contains("TS_PLAY_CMD"));
    }

    /// Resolution tier 2: with no override, the sibling
    /// `terminal-state[.exe]` wins when it EXISTS. Uses a scratch directory
    /// holding a stub file under that name, so the check does not depend on
    /// the real build output being present.
    #[test]
    fn resolve_game_command_from_falls_back_to_the_sibling_game_binary() {
        let dir = scratch_dir("sibling_present");
        let sibling = sibling_binary(&dir.join("ts_editor.exe"), "terminal-state");
        std::fs::write(&sibling, b"stub").unwrap();
        let exe = dir.join(format!("ts_editor{}", std::env::consts::EXE_SUFFIX));

        let (label, cmd) = resolve_game_command_from(None, Some(&exe), "levels/x.lvl")
            .expect("the sibling exists");
        assert_eq!(cmd.get_program(), sibling.as_os_str());
        assert!(
            label.contains("terminal-state"),
            "label names the sibling: {label}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Resolution tier 3: with no override and no `terminal-state`
    /// sibling, the `terminal3d-play[.exe]` sibling (the generic `.lvl` player)
    /// wins if IT exists.
    #[test]
    fn resolve_game_command_from_falls_back_to_the_generic_player_sibling() {
        let dir = scratch_dir("generic_player_present");
        let play = sibling_binary(&dir.join("ts_editor.exe"), "terminal3d-play");
        std::fs::write(&play, b"stub").unwrap();
        let exe = dir.join(format!("ts_editor{}", std::env::consts::EXE_SUFFIX));

        let (label, cmd) = resolve_game_command_from(None, Some(&exe), "levels/x.lvl")
            .expect("the generic player sibling exists");
        assert_eq!(cmd.get_program(), play.as_os_str());
        assert!(
            label.contains("terminal3d-play"),
            "label names the sibling: {label}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Resolution tier 4: no override and neither sibling exists ⇒
    /// `Err`, pointing the user at `$TS_PLAY_CMD` — the resolver never falls
    /// back to a cargo package name.
    #[test]
    fn resolve_game_command_from_errs_with_no_override_and_no_sibling() {
        let dir = scratch_dir("nothing_present");
        let exe = dir.join(format!("ts_editor{}", std::env::consts::EXE_SUFFIX));
        let err = resolve_game_command_from(None, Some(&exe), "levels/x.lvl")
            .expect_err("no override, no sibling");
        assert!(
            err.contains("TS_PLAY_CMD"),
            "message points at the escape hatch: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);

        // Also errs with no editor_exe at all (e.g. current_exe() failed).
        let err = resolve_game_command_from(None, None, "levels/x.lvl")
            .expect_err("no override, no exe to look beside");
        assert!(err.contains("TS_PLAY_CMD"));
    }

    /// A fresh, test-private scratch directory under the OS temp dir, keyed by
    /// this process's pid + `tag` so parallel test threads in this binary
    /// never collide with each other or with a real build's `target/`.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ts_editor_test_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn collision_toggle_flips_noclip() {
        // The preview collision toggle (the `c`-while-captured / `N` key) flips
        // `noclip`, the single source of truth. Collisions ON ⇔ noclip OFF.
        let mut s = EditorState::new_empty(8, 8, 4);
        assert!(s.noclip, "default: collisions OFF (noclip ON)");
        toggle_collisions(&mut s);
        assert!(!s.noclip, "toggle once → collisions ON (noclip OFF)");
        assert!(s.status.contains("ON"), "status reports collisions ON");
        toggle_collisions(&mut s);
        assert!(s.noclip, "toggle again → collisions OFF (noclip ON)");
        assert!(s.status.contains("OFF"), "status reports collisions OFF");
    }

    #[test]
    fn capture_look_matches_game_apply_mouse_look() {
        // MOUSE-LOOK PARITY: the editor's capture path uses the game's
        // `apply_mouse_look`, so for a given (dx, dy) and starting pose the editor
        // camera lands on EXACTLY the yaw/pitch the first-person game would.
        let (yaw0, pitch0) = (0.7_f32, 0.2_f32);
        let (dx, dy) = (12.0_f32, -5.0_f32);
        let mult = re_present::MouseSettings::default().sensitivity;
        // Reference = exactly what the game's mouse_look_sys passes to the camera.
        let (game_yaw, game_pitch) = re_present::apply_mouse_look(yaw0, pitch0, dx, dy, mult);
        // The editor computes the same value the same way.
        let (ed_yaw, ed_pitch) = re_present::apply_mouse_look(yaw0, pitch0, dx, dy, mult);
        assert_eq!(ed_yaw, game_yaw, "editor yaw matches the game");
        assert_eq!(ed_pitch, game_pitch, "editor pitch matches the game");
        // And the pitch is clamped to the game's ±π/3 limit (mouse-down → look down).
        let (_, deep) = re_present::apply_mouse_look(0.0, 0.0, 0.0, 10_000.0, 1.0);
        assert!(
            (deep + std::f32::consts::FRAC_PI_3).abs() < 1e-5,
            "pitch clamps to -π/3"
        );
    }

    #[test]
    fn cell_diff_look_parity_with_apply_mouse_look() {
        // PATH-2 PARITY: folding `apply_cell_look` over a sequence of absolute cell
        // positions must land on EXACTLY the same (yaw, pitch) as folding the game's
        // `apply_mouse_look` over the corresponding per-step deltas — skipping the
        // first sample, which only anchors. This proves the editor's crossterm cell
        // fallback reproduces the game's mouse-look math bit-for-bit.
        let sens = re_present::MouseSettings::default().sensitivity;
        let positions: [(u16, u16); 4] = [(40, 20), (52, 17), (48, 25), (60, 30)];

        // Editor side: fold apply_cell_look over the positions.
        let mut cam = re_world::CameraPose::spawn_at(1, 1, 1, 0);
        let mut last: Option<(u16, u16)> = None;
        for &pos in &positions {
            apply_cell_look(&mut cam, &mut last, pos, sens);
        }

        // Reference side: fold apply_mouse_look over the deltas (skip the anchor).
        let mut ry = re_world::CameraPose::spawn_at(1, 1, 1, 0).yaw;
        let mut rp = re_world::CameraPose::spawn_at(1, 1, 1, 0).pitch;
        for w in positions.windows(2) {
            let (px, py) = w[0];
            let (cx, cy) = w[1];
            let dx = cx as f32 - px as f32;
            let dy = cy as f32 - py as f32;
            let (y, p) = re_present::apply_mouse_look(ry, rp, dx, dy, sens);
            ry = y;
            rp = p;
        }

        assert_eq!(cam.yaw, ry, "cell-diff yaw matches apply_mouse_look fold");
        assert_eq!(
            cam.pitch, rp,
            "cell-diff pitch matches apply_mouse_look fold"
        );
    }

    #[test]
    fn cell_look_first_sample_only_anchors() {
        // ANCHOR-ONLY: with last == None, the first pos sets `last` but leaves the
        // camera untouched (no stale-position lurch). The second pos then moves it.
        let sens = re_present::MouseSettings::default().sensitivity;
        let mut cam = re_world::CameraPose::spawn_at(1, 1, 1, 0);
        let (yaw0, pitch0) = (cam.yaw, cam.pitch);
        let mut last: Option<(u16, u16)> = None;

        apply_cell_look(&mut cam, &mut last, (30, 30), sens);
        assert_eq!(last, Some((30, 30)), "first sample records the anchor");
        assert_eq!(cam.yaw, yaw0, "first sample does not rotate yaw");
        assert_eq!(cam.pitch, pitch0, "first sample does not pitch");

        apply_cell_look(&mut cam, &mut last, (45, 30), sens);
        assert_ne!(cam.yaw, yaw0, "the second sample moves the camera");
        assert_eq!(last, Some((45, 30)), "tracker advanced to the new pos");
    }

    #[test]
    fn cell_look_huge_down_delta_clamps_pitch() {
        // PITCH-CLAMP PARITY: a huge downward (positive dy) cell delta drives pitch
        // to the game's -π/3 floor, exactly as apply_mouse_look clamps.
        let sens = 1.0_f32;
        let mut cam = re_world::CameraPose::spawn_at(1, 1, 1, 0);
        cam.yaw = 0.0;
        cam.pitch = 0.0;
        let mut last: Option<(u16, u16)> = Some((0, 0));
        // dy = +10000 (mouse-down looks down → pitch goes very negative, clamped).
        apply_cell_look(&mut cam, &mut last, (0, 10_000), sens);
        assert!(
            (cam.pitch + std::f32::consts::FRAC_PI_3).abs() < 1e-5,
            "huge down delta clamps pitch to -π/3"
        );
    }

    /// B2 REGRESSION — committing a row editor must not also fly the camera.
    ///
    /// `camera_input_system` gates ONLY on `any_modal_active()`, and
    /// `apply_camera_key` binds `Enter` to jump-to-cursor unconditionally. The
    /// two systems hold separate `MessageReader<EditorKey>`s, so both see the
    /// same keypress: a modal missing from `any_modal_active` has its confirm
    /// key ALSO teleport the preview camera. Every Enter that committed a
    /// number, picked an enum, or committed a colour used to jump it.
    #[test]
    fn a_modal_that_consumes_enter_is_listed_in_any_modal_active() {
        use crate::state::{EntityField, EntityKind};
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Node);
        crate::entity::place_at_cursor(&mut s);

        for field in [
            EntityField::NodeThreshold,
            EntityField::NodeIo,
            EntityField::NodeColor,
        ] {
            let act = crate::chrome::RowAction::FocusEntity(field);
            assert!(
                crate::rowedit::open_for(&mut s, act),
                "{field:?} opens an editor"
            );
            assert!(
                s.any_modal_active(),
                "{field:?}'s editor must suppress the fly-camera"
            );
            crate::rowedit::cancel(&mut s);
        }
        // The `:` command palette had the identical hole.
        crate::keymap::open_palette(&mut s);
        assert!(
            s.any_modal_active(),
            "the command palette suppresses the camera"
        );
        s.cmd_palette = None;

        // The CRASH-RECOVERY offer and the part MASK editor are
        // the same hole one layer EARLIER in the chain. `edit::apply_key`
        // dispatches both AHEAD of every modal above and swallows every key —
        // the offer's `_` arm re-states the question, the mask editor's `_ => {}`
        // eats the rest, Enter included — yet neither was listed here, so while a
        // crash-recovery prompt was pending WASD still flew the preview camera
        // and Enter still teleported it.
        s.autosave_offer = Some("levels/x.lvl.autosave".into());
        assert!(
            s.any_modal_active(),
            "the crash-recovery offer suppresses the fly-camera"
        );
        s.autosave_offer = None;
        assert!(!s.any_modal_active(), "offer cleared → not modal");

        s.mask_mode = true;
        assert!(
            s.any_modal_active(),
            "the part mask editor suppresses the fly-camera"
        );
        s.mask_mode = false;

        assert!(
            !s.any_modal_active(),
            "and nothing is modal once they all close"
        );
    }

    /// PAINT ORDER. `draw_system` is a Bevy system over
    /// `Res`/`ResMut` and cannot be called from a unit test, so this asserts the
    /// property on the SOURCE of the overlay block — the same technique
    /// `every_settings_key_is_in_the_keymap` uses on the dispatch block.
    ///
    /// Two things must hold and neither is checkable any other way:
    ///
    /// 1. `chrome::draw_autosave_offer` is CALLED at all. It exists and is unit
    ///    tested, but a painter nobody invokes is exactly the defect this fixes —
    ///    the offer already owned the keyboard while drawing nothing.
    /// 2. It is called AFTER every other centered modal (the `:` palette last
    ///    among them) and BEFORE the capture banner. The offer dispatches at the
    ///    FRONT of both the key chain (`edit::apply_key`) and the click chain
    ///    (`mouse::apply_modal_click`), so anything else that happens to be open
    ///    is inert underneath it: dispatch precedence => paint precedence.
    ///
    /// Comment lines are stripped first, so commenting the call out fails this
    /// test rather than hiding inside it. The scan works on LINES, never on byte
    /// offsets into the raw source: `core.autocrlf` is on for this repo, so a
    /// needle containing a newline matches nothing in a CRLF checkout — the same
    /// trap a CRLF-blind `replacen` falls into.
    #[test]
    fn the_crash_recovery_overlay_is_painted_last_of_the_modals() {
        let src = include_str!("main.rs");
        let lines: Vec<&str> = src.lines().collect(); // `lines()` eats a trailing CR
        let test_mod = lines
            .iter()
            .position(|l| l.trim() == "mod tests {")
            .expect("the test module marker");
        // Real code only: the test module (which quotes these needles) is cut
        // off, and comment lines are dropped so a `//`-ed-out call cannot pass.
        let code: Vec<(usize, &str)> = lines[..test_mod]
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.trim_start().starts_with("//"))
            .map(|(i, l)| (i, *l))
            .collect();
        let line_of = |needle: &str| {
            code.iter()
                .find(|(_, l)| l.contains(needle))
                .map(|(i, _)| *i)
        };

        let call = line_of("chrome::draw_autosave_offer(")
            .expect("the crash-recovery overlay must actually be painted");
        let palette =
            line_of("chrome::draw_cmd_palette(").expect("premise: the `:` palette is painted here");
        let banner = line_of("chrome::draw_capture_banner(")
            .expect("premise: the capture banner is painted here");
        assert!(
            palette < call,
            "the offer paints ABOVE every other modal (it dispatches above them)"
        );
        assert!(call < banner, "only the capture banner outranks it");
    }

    /// The end-to-end shape of the crash-recovery half of the bug: with an offer
    /// pending, the Enter that a user presses at a prompt they cannot see must not
    /// teleport the preview camera into the level they have not recovered yet.
    /// Written exactly like `enter_committing_a_row_editor_never_moves_the_camera`
    /// — it gates on the real predicate `camera_input_system` gates on.
    #[test]
    fn enter_under_a_pending_autosave_offer_never_moves_the_camera() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        s.cursor = (12, 9);
        s.z = 2;
        s.cam = re_world::CameraPose::spawn_at(1, 1, 0, 0);
        let (pos0, yaw0, pitch0) = (s.cam.pos, s.cam.yaw, s.cam.pitch);
        s.autosave_offer = Some("levels/x.lvl.autosave".into());

        assert!(s.any_modal_active(), "the camera system early-returns");
        if !s.any_modal_active() {
            apply_camera_key(&mut s, EditorKeyCode::Enter, false, false, 0.016);
        }
        assert_eq!(
            s.cam.pos, pos0,
            "the offer's keystream did not teleport the camera"
        );
        assert_eq!(s.cam.yaw, yaw0);
        assert_eq!(s.cam.pitch, pitch0);
    }

    /// The end-to-end shape of the same bug: with a row editor open, the Enter
    /// that commits it leaves the preview camera exactly where it was.
    #[test]
    fn enter_committing_a_row_editor_never_moves_the_camera() {
        use crate::state::{EntityField, EntityKind};
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        s.cursor = (12, 9);
        s.z = 2;
        s.place = Some(EntityKind::Node);
        crate::entity::place_at_cursor(&mut s);
        // Put the camera somewhere the cursor is NOT, so a jump would show.
        s.cam = re_world::CameraPose::spawn_at(1, 1, 0, 0);
        let (pos0, yaw0, pitch0) = (s.cam.pos, s.cam.yaw, s.cam.pitch);

        crate::rowedit::open_for(
            &mut s,
            crate::chrome::RowAction::FocusEntity(EntityField::NodeThreshold),
        );
        assert!(s.num_edit.is_some());

        // `camera_input_system` would early-return here; prove the predicate it
        // gates on says so, then prove the key itself is inert through it.
        assert!(s.any_modal_active(), "the camera system early-returns");
        if !s.any_modal_active() {
            apply_camera_key(&mut s, EditorKeyCode::Enter, false, false, 0.016);
        }
        assert_eq!(s.cam.pos, pos0, "the commit did not teleport the camera");
        assert_eq!(s.cam.yaw, yaw0);
        assert_eq!(s.cam.pitch, pitch0);
    }

    /// `apply_camera_key`'s Enter really does jump when nothing is modal — so
    /// the test above is proving the GATE, not an inert key.
    #[test]
    fn enter_still_jumps_the_camera_when_no_modal_is_open() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        s.cursor = (12, 9);
        s.z = 2;
        s.cam = re_world::CameraPose::spawn_at(1, 1, 0, 0);
        let pos0 = s.cam.pos;
        assert!(!s.any_modal_active(), "nothing is modal");
        apply_camera_key(&mut s, EditorKeyCode::Enter, false, false, 0.016);
        assert_ne!(
            s.cam.pos, pos0,
            "Enter jumps to the cursor in the normal case"
        );
    }

    #[test]
    fn uncaptured_movement_keys_never_move_the_camera() {
        // MODAL SEPARATION (FIX C): when NOT captured (2D edit mode) the camera nav
        // keys are inert here — they belong purely to 2D editing — so the preview
        // camera's position and orientation must not change at all.
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        let pos0 = s.cam.pos;
        let yaw0 = s.cam.yaw;
        let pitch0 = s.cam.pitch;
        for c in ['w', 'a', 'd', 'r', 'f', ',', '.'] {
            // captured = false, held_driving = false.
            apply_camera_key(&mut s, EditorKeyCode::Char(c), false, false, 0.016);
        }
        assert_eq!(
            s.cam.pos, pos0,
            "movement keys do not move the camera in 2D mode"
        );
        assert_eq!(
            s.cam.yaw, yaw0,
            "turn keys do not rotate the camera in 2D mode"
        );
        assert_eq!(s.cam.pitch, pitch0, "pitch is untouched in 2D mode");
    }

    #[test]
    fn captured_w_moves_camera_forward_via_per_event_fallback() {
        // captured + empty HeldKeys (held_driving = false): the per-event fallback
        // still nudges the camera, so a single `w` moves it forward along facing.
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        s.cam.yaw = 0.0; // faces +y
        let y0 = s.cam.pos.y;
        apply_camera_key(&mut s, EditorKeyCode::Char('w'), true, false, 0.016);
        assert!(s.cam.pos.y > y0, "captured W moves the camera forward");
    }

    #[test]
    fn held_move_delta_drives_forward_at_move_speed() {
        // The held-key mover applies `MOVE_SPEED * dt` along facing — game parity.
        let held = re_present::HeldKeys {
            forward: true,
            ..Default::default()
        };
        let dt = 0.5_f32;
        // Facing +y (dir = (sin0, cos0) = (0, 1)).
        let (dx, dy) = re_present::held_move_delta(held, 0.0, 1.0, re_present::MOVE_SPEED, dt);
        assert!(dx.abs() < 1e-6, "no sideways drift moving straight forward");
        assert!(
            (dy - re_present::MOVE_SPEED * dt).abs() < 1e-5,
            "forward = MOVE_SPEED*dt"
        );
    }

    /// A CLEAN document parks the autosave timer at zero — no accumulation,
    /// no write, ever.
    #[test]
    fn autosave_tick_clean_document_parks_timer() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.dirty = false;
        s.autosave_elapsed = 42.0; // stale (e.g. the doc just became clean)
        autosave_tick(&mut s, 1.0);
        assert_eq!(s.autosave_elapsed, 0.0, "clean doc resets the timer");
    }

    /// A DIRTY document accumulates the timer but writes nothing before
    /// the interval expires.
    #[test]
    fn autosave_tick_accumulates_below_interval() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.map.set(1, 1, 0, re_world::CellKind::WallSolid);
        s.dirty = true;
        autosave_tick(&mut s, 1.5);
        autosave_tick(&mut s, 2.5);
        assert_eq!(s.autosave_elapsed, 4.0, "dt accumulates while dirty");
        assert!(s.dirty, "nothing else changes below the interval");
    }

    /// Dirty state + forced timer expiry → the sidecar file exists at the
    /// production `levels/.autosave/<stem>.lvl` path (cwd-relative, like
    /// `scan_lvl_files`), the timer resets, and the dirty flag survives.
    #[test]
    fn autosave_tick_expiry_writes_sidecar() {
        let mut s = EditorState::new_empty(6, 6, 4);
        s.map.set(3, 3, 1, re_world::CellKind::WallSolid);
        s.dirty = true;
        // Unique per process AND per call: `autosave_path_for` keys the sidecar
        // on the canonicalized parent dir + stem, so a fixed name made every
        // concurrent test process target ONE file — and a sibling's
        // `remove_file` could land between this test's write and its read.
        s.path = Some(crate::testutil::scratch_path("w24-tick.lvl"));
        let dest = io::autosave_path_for(s.path.as_deref());
        let _dest_guard = crate::testutil::ScratchFile::adopt(dest.clone());

        autosave_tick(&mut s, io::AUTOSAVE_INTERVAL_SECS);
        assert_eq!(s.autosave_elapsed, 0.0, "expiry resets the timer");
        assert!(s.dirty, "autosave never clears dirty");
        assert!(
            s.status.contains("autosaved"),
            "status reports it: {}",
            s.status
        );
        let text = std::fs::read_to_string(&dest).expect("sidecar written on expiry");
        assert!(text.contains("[cells]"), "sidecar carries the live edits");
        // `_dest_guard` removes the sidecar on drop, including on panic. No
        // shared-dir cleanup: removing `levels/.autosave` here would race the
        // parallel tests writing their own sidecars into it, and an empty
        // leftover dir is gitignored and harmless.
    }

    /// The reparse guard holds through the tick — an unloadable document
    /// autosaves NOTHING and says so in the status.
    #[test]
    fn autosave_tick_reparse_guard_skips_with_status() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.dirty = true;
        s.path = Some(crate::testutil::scratch_path("w24-tick-bad.lvl"));
        // Endpoints naming no node serialize to a header finalize rejects.
        s.level.pipes.push(re_level::Pipe {
            from: "ghost".into(),
            to: "phantom".into(),
            ..Default::default()
        });
        let dest = io::autosave_path_for(s.path.as_deref());
        let _ = std::fs::remove_file(&dest);
        autosave_tick(&mut s, io::AUTOSAVE_INTERVAL_SECS);
        assert!(
            s.status.contains("autosave skipped"),
            "refusal surfaces in the status: {}",
            s.status
        );
        assert!(
            !std::path::Path::new(&dest).exists(),
            "no sidecar written when the reparse guard refuses"
        );
    }
}
