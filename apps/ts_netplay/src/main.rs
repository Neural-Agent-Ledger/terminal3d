//! `terminal-state-netplay` — the TERMINAL STATE multiplayer testbed.
//!
//! A runnable server+client testbed for evaluating Bevy replication (bevy_replicon +
//! bevy_replicon_renet/UDP) with server-authoritative shooting and distance-accurate
//! ASCII bullet rendering.
//!
//! # Usage
//! ```text
//! terminal-state-netplay --server [--port 5000]
//! terminal-state-netplay --client [--addr 127.0.0.1] [--port 5000]
//! terminal-state-netplay --help
//! ```
//!
//! ## Server-first ordering
//!
//! **Start the `--server` first, then one or more `--client` instances.** The
//! client's renet handshake will silently retry against a not-yet-listening
//! server, so a client launched before the server simply never connects.
//!
//! # Architecture
//! - `sim` — server-authoritative ECS systems (movement, fire, bullet integrate/hit).
//!   Pure compute; the SERVER app adds them, the client never does.
//! - `transport` — renet/UDP socket + transport setup (server + client).
//! - `client` — input → `MoveIntent`/`FireInput`, render replicated world+bullets,
//!   present via `re_present` (the ONLY terminal touch point).
//!
//! `ts_netplay` is the only crate that depends on `re_present`; `ts_net` stays
//! terminal-free.
//!
//! This binary is an experimental testbed, isolated from the shipped game
//! build (see the `no_net_in_game_build` guard test in `tests/`).

use std::net::Ipv4Addr;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy::state::app::StatesPlugin;
use bevy::MinimalPlugins;
use bevy_replicon::prelude::*;
use bevy_replicon_renet::RepliconRenetPlugins;
use glam::Vec3;
use re_core::Framebuffer;
use re_world::{CameraPose, Map};
use ts_netplay::{client, sim, transport};

/// The fixed arena `.lvl` the testbed loads (a simple open room). Falls back to
/// `re_world::test_room()` if the file cannot be loaded.
const ARENA_LEVEL: &str = "levels/01.lvl";

/// Default terminal grid the framebuffer starts at (the present backend resizes
/// it to the real terminal each frame).
const DEFAULT_COLS: usize = 120;
const DEFAULT_ROWS: usize = 40;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = Cli::parse(&args);

    if cfg.help {
        print_help();
        return;
    }

    match cfg.mode {
        Mode::Server => run_server(cfg.port),
        Mode::Client => run_client(cfg.addr, cfg.port),
    }
}

// ── arena ────────────────────────────────────────────────────────────────────

/// Load the fixed arena map, falling back to the built-in test room on any error
/// (a missing/invalid `.lvl` must never crash the testbed). Returns the map and a
/// spawn point inside its open floor.
fn load_arena() -> (Map, sim::SpawnPoint) {
    match re_level::load(ARENA_LEVEL) {
        Ok(level) => {
            let map = level.build_map();
            let p = &level.player;
            // Cell-centre spawn at the level's [player] cell; yaw 0 faces +y.
            let spawn = sim::SpawnPoint {
                pos: Vec3::new(p.x as f32 + 0.5, p.y as f32 + 0.5, p.z as f32 + 0.5),
                yaw: 0.0,
            };
            (map, spawn)
        }
        Err(err) => {
            eprintln!("ts_netplay: failed to load {ARENA_LEVEL} ({err}); using test_room");
            (re_world::test_room(), sim::SpawnPoint::default())
        }
    }
}

// ── server ───────────────────────────────────────────────────────────────────

/// Build and run the headless server App.
fn run_server(port: u16) {
    let (map, spawn) = load_arena();

    let mut app = App::new();
    app.add_plugins(
        MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        ))),
    )
    .add_plugins(StatesPlugin)
    .add_plugins(RepliconPlugins.set(ServerPlugin::new(PostUpdate)))
    .add_plugins(RepliconRenetPlugins)
    .add_plugins(ts_net::NetPlugin);

    app.insert_resource(map).insert_resource(spawn);
    sim::add_server_sim(&mut app);

    // Channels must be frozen before reading RepliconChannels for renet.
    app.finish();
    app.cleanup();
    let bound = match transport::insert_server(&mut app, port, 32) {
        Ok(bound) => bound,
        Err(err) => {
            eprintln!("terminal-state-netplay: could not start server on port {port}: {err}");
            std::process::exit(1);
        }
    };

    println!("terminal-state-netplay: SERVER listening on 0.0.0.0:{bound}");
    println!("Start the --server first, then one or more --client instances.");
    app.run();
}

// ── client ───────────────────────────────────────────────────────────────────

/// Build and run the client App (terminal present + input + render).
fn run_client(addr: Ipv4Addr, port: u16) {
    let (map, spawn) = load_arena();

    let mut app = App::new();
    app.add_plugins(
        MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        ))),
    )
    .add_plugins(StatesPlugin)
    // Server + client build the SAME RepliconPlugins set (incl. ServerPlugin) so
    // both apps register the identical replication channels/rules in the identical
    // schedule (PostUpdate). The channel set must match byte-for-byte on both ends
    // of the renet connection or the handshake's channel negotiation fails; the
    // unused server half is inert on the client (no RenetServer resource is added).
    .add_plugins(RepliconPlugins.set(ServerPlugin::new(PostUpdate)))
    .add_plugins(RepliconRenetPlugins)
    .add_plugins(ts_net::NetPlugin)
    // The terminal backend (present + input). ts_netplay is the ONLY crate that
    // may add this.
    .add_plugins(re_present::PresentPlugin);

    // Resources the render + present read each frame.
    app.insert_resource(map)
        .insert_resource(spawn)
        .insert_resource(Framebuffer::new(DEFAULT_COLS, DEFAULT_ROWS))
        .insert_resource(client::RenderFov(re_render::Fov::default()))
        // Insert-if-absent: PresentPlugin inits most, but be defensive.
        .insert_resource(re_present::ColorMode::default())
        .insert_resource(re_present::MouseSettings::default());

    // The local player camera the present backend moves from held keys; also the
    // aim source for FireInput.
    app.world_mut().spawn(CameraPose {
        pos: spawn.pos,
        yaw: spawn.yaw,
        pitch: 0.0,
    });

    client::add_client_systems(&mut app);

    app.finish();
    app.cleanup();
    if let Err(err) = transport::insert_client(&mut app, client_id_from_pid(), addr, port) {
        eprintln!("terminal-state-netplay: could not set up client for {addr}:{port}: {err}");
        std::process::exit(1);
    }

    println!("terminal-state-netplay: CLIENT connecting to {addr}:{port}");
    app.run();
}

/// A best-effort unique client id for this process (the OS pid). The renet
/// netcode only needs distinct ids per connection within a session.
fn client_id_from_pid() -> u64 {
    std::process::id() as u64
}

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Server,
    Client,
}

#[derive(Clone, Debug)]
struct Cli {
    mode: Mode,
    addr: Ipv4Addr,
    port: u16,
    help: bool,
}

impl Cli {
    fn parse(args: &[String]) -> Self {
        let help = args.iter().any(|a| a == "--help" || a == "-h");
        let mode = if args.iter().any(|a| a == "--server") {
            Mode::Server
        } else {
            Mode::Client // default
        };
        let port: u16 = find_flag(args, "--port")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000);
        let addr: Ipv4Addr = find_flag(args, "--addr")
            .and_then(|v| v.parse().ok())
            .unwrap_or(Ipv4Addr::LOCALHOST);
        Cli {
            mode,
            addr,
            port,
            help,
        }
    }
}

fn find_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn print_help() {
    println!("terminal-state-netplay -- TERMINAL STATE multiplayer replication testbed");
    println!();
    println!("USAGE:");
    println!("    terminal-state-netplay --server [--port <N>]");
    println!("    terminal-state-netplay --client [--addr <IP>] [--port <N>]");
    println!("    terminal-state-netplay --help");
    println!();
    println!("FLAGS:");
    println!("    --server          Run the authoritative server (binds 0.0.0.0:<port>).");
    println!("    --client          Run a terminal client (the default if no mode given).");
    println!("    --addr <IP>       Server address to connect to (client). Default 127.0.0.1.");
    println!("    --port <N>        UDP port. Default 5000.");
    println!("    -h, --help        Print this help and exit.");
    println!();
    println!("ORDERING:");
    println!("    Start the --server first, then one or more --client instances.");
    println!("    A client launched before the server will not connect.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_client_localhost_5000() {
        let cli = Cli::parse(&["bin".into()]);
        assert_eq!(cli.mode, Mode::Client);
        assert_eq!(cli.addr, Ipv4Addr::LOCALHOST);
        assert_eq!(cli.port, 5000);
        assert!(!cli.help);
    }

    #[test]
    fn server_flag_selects_server() {
        let cli = Cli::parse(&["bin".into(), "--server".into()]);
        assert_eq!(cli.mode, Mode::Server);
    }

    #[test]
    fn parses_addr_and_port() {
        let cli = Cli::parse(&[
            "bin".into(),
            "--client".into(),
            "--addr".into(),
            "10.0.0.5".into(),
            "--port".into(),
            "7777".into(),
        ]);
        assert_eq!(cli.mode, Mode::Client);
        assert_eq!(cli.addr, Ipv4Addr::new(10, 0, 0, 5));
        assert_eq!(cli.port, 7777);
    }

    #[test]
    fn help_flag_detected() {
        assert!(Cli::parse(&["bin".into(), "--help".into()]).help);
        assert!(Cli::parse(&["bin".into(), "-h".into()]).help);
    }

    #[test]
    fn arena_loads_or_falls_back() {
        // Must never panic: either the .lvl loads or test_room is used.
        let (map, _spawn) = load_arena();
        let (sx, sy, sz) = map.size();
        assert!(sx > 0 && sy > 0 && sz > 0);
    }
}
