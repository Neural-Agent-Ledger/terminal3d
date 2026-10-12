//! `re_render` performance harness — stage-by-stage timing of the composite
//! render pipeline, mirroring `composite_render_sys` exactly.
//!
//! # Usage
//!
//! Release (the only meaningful number):
//!   cargo run --release -p re_render --example perfbench -- [OPTIONS]
//!
//! Debug (for debug-vs-release gap measurement only):
//!   cargo run -p re_render --example perfbench -- [OPTIONS]
//!
//! # Options (all have defaults)
//!   --level <path>         Level file to load    [default: levels/01.lvl]
//!   --complex-level <path> Complex-walls level   [default: levels/101-facade-skyline-courtyard.lvl]
//!   --fractal-level <path> Fractal portal level  [default: levels/92-fractal-portal.lvl]
//!   --warp-level <path>    Warp-portal geo level [default: levels/93-warp-portal.lvl]
//!   --cols  <N>            Terminal columns       [default: 240]
//!   --rows  <N>            Terminal rows          [default: 64]
//!   --scenario-cols <N>    Cols for new rotation/translation/mixed scenarios [default: 280]
//!   --scenario-rows <N>    Rows for new rotation/translation/mixed scenarios [default: 72]
//!   --frames <N>           Measured frames        [default: 90]
//!   --warmup <N>           Warmup frames skipped  [default: 10]
//!   --skip-classic         Skip scenarios A-E (only run F/G/H/I/J/K)
//!   --skip-motion          Skip scenarios F/G/H (only run A-E/I/J/K)
//!   --skip-fractal         Skip scenarios I/J (fractal portal + tunnel microbench)
//!   --skip-warp            Skip scenario K (warp-portal geometry perf gate)
//!
//! Scenarios A-E (classic), F/G/H (motion), I (fractal portal + animated-idle),
//! J (render_tunnel full-fill microbench), and K (warp-portal geometry) are run
//! sequentially and printed as tables of per-stage avg/max µs + total FPS.
//!
//! # What is measured
//!
//! The harness exercises the composite pipeline stage-for-stage:
//!   FAR   — FarCache::rebake (when needed) + fill_far_layer
//!   NEAR  — build_world_layers_fov_into (gated by dirty-flag / NearKey)
//!   SUBCELL — refine_subcells (local 2×2 silhouette anti-alias)
//!   EMIT  — tick_emitters + splat_particles
//!   RESOLVE — CompositeStack::resolve into the Framebuffer
//!
//! An IDLE frame's FAR fill + EMIT + RESOLVE still run (cheap), while NEAR +
//! SUBCELL are skipped (0 µs) once the NearKey stabilises.
//!
//! # New scenarios (F/G/H) — rotation/translation baseline for panoramic cache
//!
//! These run at --scenario-cols/--scenario-rows (default 280x72, fullscreen-ish)
//! on the --complex-level (default 101-facade-skyline-courtyard — a 22x22x9
//! showcase courtyard: an authored `[sky]`, 5 materials, 2 zones. Deliberately
//! not a campaign level: this crate's tests/examples must not depend on
//! campaign content, and this is the densest non-campaign showcase level
//! available) over 120 frames after warmup.
//!
//!   PURE_ROTATION (F): camera POSITION fixed, yaw sweeps +0.03 rad/frame,
//!     optional small pitch oscillation. Yaw bucket width is TAU/128 ≈ 0.049 rad,
//!     so 0.03 rad/frame crosses a bucket every ~1.6 frames → near re-casts almost
//!     every frame. THIS is the baseline the panoramic cache will make nearly free.
//!     near µs ≈ full cast, near cache hits ≈ 0.
//!
//!   TRANSLATION (G): camera walks forward ~0.1 cell/frame, yaw fixed.
//!     Cell boundary every ~10 frames → re-casts ~12x per 120-frame run.
//!     Parallax case — panoramic cache will still require a re-cast (or
//!     column-shift approximation) on each cell crossing.
//!
//!   MIXED (H): walk ~0.05 cell/frame + yaw +0.02 rad/frame simultaneously.
//!     Both position and yaw change every frame → near re-cast most frames.
//!
//! # Env toggles
//!   PERFBENCH_ONESHOT=1   FAR re-bakes the WHOLE 480-col ring in one call every
//!                         time the tier policy fires (the pre-incremental BEFORE
//!                         baseline — the ~30 ms worst-frame spike). Unset (default)
//!                         uses the INCREMENTAL chunked bake (~one chunk/frame).

use std::env;
use std::time::Instant;

use re_core::{CompositeStack, Framebuffer};
use re_render::{
    ascii_shader::MaterialRegistry,
    build_world_layers_fov_capture_into,
    composite::{refine_subcells_buf, ANIM_SELECTIVE_MAX_FRAC},
    demo_emitters,
    emitter::{splat_particles, tick_emitters, ParticlePool, ParticleRng},
    far_cache::FarCache,
    gbuffer::NearGBuffer,
    near_cache::NearRing,
    registry_for_map, render_tunnel, reshade_animated_into,
    tier::{far_needs_rebake, TierClock},
    view_params, FarFreeze, Fov, TunnelParams,
};
use re_world::CameraPose;

/// Build a registry whose 8 fixed slots (0..7) are all STATIC catalog materials
/// (so `map_local_any_animated` returns false), then APPEND the full catalog by
/// name exactly like `registry_for_map` — so the registry-GLOBAL `any_animated`
/// stays true (the catalog animates) while the map-local near check is false.
/// This isolates the near-cache idle win the fix unlocks, independent of level
/// 01's authored-animated circuit/matrix walls.
fn static_surface_registry() -> MaterialRegistry {
    let mut r = MaterialRegistry::new();
    // "stone" is a static catalog material; use it for all 8 fixed slots.
    let stone = MaterialRegistry::catalog_material("stone").expect("stone in catalog");
    for _ in 0..8 {
        r.register_named("stone", stone.clone());
    }
    for name in re_render::ascii_shader::ids::CATALOG_NAMES {
        if let Some(m) = MaterialRegistry::catalog_material(name) {
            r.register_named(name, m);
        }
    }
    r
}

// ─── CLI parsing ─────────────────────────────────────────────────────────────

struct Config {
    level: String,
    complex_level: String,
    /// Fractal-portal demo level for scenarios I/J.
    fractal_level: String,
    /// Warp-portal geometry demo level for scenario K.
    warp_level: String,
    cols: usize,
    rows: usize,
    /// Grid for the new rotation/translation/mixed scenarios (F/G/H).
    scenario_cols: usize,
    scenario_rows: usize,
    frames: usize,
    warmup: usize,
    skip_classic: bool,
    skip_motion: bool,
    /// Skip the fractal-portal and tunnel micro-benchmarks (scenarios I/J).
    skip_fractal: bool,
    /// Skip the warp-portal geometry perf gate (scenario K).
    skip_warp: bool,
    /// Run ONLY the big-world procedural-brickmap baseline (scenario BW). Selected
    /// by the positional `bigworld` arg; when set, all other scenarios are skipped
    /// so `-- bigworld` reports just the octree baseline.
    bigworld: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            level: "levels/01.lvl".to_string(),
            complex_level: "levels/101-facade-skyline-courtyard.lvl".to_string(),
            fractal_level: "levels/92-fractal-portal.lvl".to_string(),
            warp_level: "levels/93-warp-portal.lvl".to_string(),
            cols: 240,
            rows: 64,
            scenario_cols: 280,
            scenario_rows: 72,
            frames: 90,
            warmup: 10,
            skip_classic: false,
            skip_motion: false,
            skip_fractal: false,
            skip_warp: false,
            bigworld: false,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--level" if i + 1 < args.len() => {
                cfg.level = args[i + 1].clone();
                i += 2;
            }
            "--complex-level" if i + 1 < args.len() => {
                cfg.complex_level = args[i + 1].clone();
                i += 2;
            }
            "--cols" if i + 1 < args.len() => {
                cfg.cols = args[i + 1].parse().expect("--cols must be a number");
                i += 2;
            }
            "--rows" if i + 1 < args.len() => {
                cfg.rows = args[i + 1].parse().expect("--rows must be a number");
                i += 2;
            }
            "--scenario-cols" if i + 1 < args.len() => {
                cfg.scenario_cols = args[i + 1]
                    .parse()
                    .expect("--scenario-cols must be a number");
                i += 2;
            }
            "--scenario-rows" if i + 1 < args.len() => {
                cfg.scenario_rows = args[i + 1]
                    .parse()
                    .expect("--scenario-rows must be a number");
                i += 2;
            }
            "--frames" if i + 1 < args.len() => {
                cfg.frames = args[i + 1].parse().expect("--frames must be a number");
                i += 2;
            }
            "--warmup" if i + 1 < args.len() => {
                cfg.warmup = args[i + 1].parse().expect("--warmup must be a number");
                i += 2;
            }
            "--fractal-level" if i + 1 < args.len() => {
                cfg.fractal_level = args[i + 1].clone();
                i += 2;
            }
            "--warp-level" if i + 1 < args.len() => {
                cfg.warp_level = args[i + 1].clone();
                i += 2;
            }
            "--skip-classic" => {
                cfg.skip_classic = true;
                i += 1;
            }
            "--skip-motion" => {
                cfg.skip_motion = true;
                i += 1;
            }
            "--skip-fractal" => {
                cfg.skip_fractal = true;
                i += 1;
            }
            "--skip-warp" => {
                cfg.skip_warp = true;
                i += 1;
            }
            "bigworld" => {
                // Positional scenario selector: run ONLY the big-world baseline.
                cfg.bigworld = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown arg: {other} (ignored)");
                i += 1;
            }
        }
    }
    cfg
}

// ─── Per-stage timing accumulator ────────────────────────────────────────────

#[derive(Default, Clone)]
struct Timings {
    far_us: Vec<f64>,
    near_us: Vec<f64>,
    subcell_us: Vec<f64>,
    emit_us: Vec<f64>,
    resolve_us: Vec<f64>,
    total_us: Vec<f64>,
    /// How many frames had a near cache HIT (near_us == 0 after warmup).
    near_hits: usize,
    /// How many frames triggered a far re-bake.
    far_rebakes: usize,
    /// How many frames ran the SELECTIVE re-shade (cheap animated-cells-only path).
    selective_frames: usize,
    /// Animated-cell fraction of the grid at the last full capture (0..1).
    anim_frac: f64,
    /// Total measured frames.
    n: usize,
}

impl Timings {
    fn push(
        &mut self,
        far: f64,
        near: f64,
        subcell: f64,
        emit: f64,
        resolve: f64,
        far_rebaked: bool,
    ) {
        let tot = far + near + subcell + emit + resolve;
        self.far_us.push(far);
        self.near_us.push(near);
        self.subcell_us.push(subcell);
        self.emit_us.push(emit);
        self.resolve_us.push(resolve);
        self.total_us.push(tot);
        if near == 0.0 {
            self.near_hits += 1;
        }
        if far_rebaked {
            self.far_rebakes += 1;
        }
        self.n += 1;
    }

    fn summarize(samples: &[f64]) -> (f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0);
        }
        let mut s = samples.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg = s.iter().sum::<f64>() / s.len() as f64;
        let max = *s.last().unwrap();
        (avg, max)
    }

    fn print_table(&self, label: &str, n: usize) {
        let (far_avg, far_max) = Self::summarize(&self.far_us);
        let (near_avg, near_max) = Self::summarize(&self.near_us);
        let (sub_avg, sub_max) = Self::summarize(&self.subcell_us);
        let (emit_avg, emit_max) = Self::summarize(&self.emit_us);
        let (res_avg, res_max) = Self::summarize(&self.resolve_us);
        let (tot_avg, tot_max) = Self::summarize(&self.total_us);
        let fps = if tot_avg > 0.0 {
            1_000_000.0 / tot_avg
        } else {
            f64::INFINITY
        };

        println!();
        println!("=== {label} ({n} frames after warmup) ===");
        println!("{:<30} {:>10} {:>10}", "Stage", "avg µs", "max µs");
        println!("{}", "-".repeat(52));
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "FAR (fill + rebake when dirty)", far_avg, far_max
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "NEAR (raycast near layers)", near_avg, near_max
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "SUBCELL (refine_subcells)", sub_avg, sub_max
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "EMIT (tick + splat)", emit_avg, emit_max
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "RESOLVE (fold layers)", res_avg, res_max
        );
        println!("{}", "-".repeat(52));
        println!("{:<30} {:>10.1} {:>10.1}", "TOTAL", tot_avg, tot_max);
        println!("Implied FPS (1e6/tot_avg):  {fps:.1}");
        println!(
            "Near cache HITs: {}/{n}  ({:.0}% idle savings)",
            self.near_hits,
            100.0 * self.near_hits as f64 / n.max(1) as f64
        );
        println!("Far re-bakes: {}/{n}", self.far_rebakes);
    }

    /// Extended table used by the new rotation/translation scenarios. Adds a
    /// "near_us when MISS only" column and the near-miss count alongside hits.
    fn print_motion_table(&self, label: &str, n: usize) {
        let (far_avg, far_max) = Self::summarize(&self.far_us);
        // Near avg over ALL frames (many are 0 = cache hits).
        let (near_avg_all, near_max) = Self::summarize(&self.near_us);
        // Near avg over MISS frames only (exclude the zeros).
        let miss_samples: Vec<f64> = self.near_us.iter().copied().filter(|&v| v > 0.0).collect();
        let near_misses = miss_samples.len();
        let (near_avg_miss, _) = Self::summarize(&miss_samples);
        let (sub_avg, sub_max) = Self::summarize(&self.subcell_us);
        let (emit_avg, emit_max) = Self::summarize(&self.emit_us);
        let (res_avg, res_max) = Self::summarize(&self.resolve_us);
        let (tot_avg, tot_max) = Self::summarize(&self.total_us);
        let fps = if tot_avg > 0.0 {
            1_000_000.0 / tot_avg
        } else {
            f64::INFINITY
        };

        println!();
        println!("=== {label} ({n} frames after warmup) ===");
        println!("{:<30} {:>10} {:>10}", "Stage", "avg µs", "max µs");
        println!("{}", "-".repeat(52));
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "FAR (fill + rebake when dirty)", far_avg, far_max
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "NEAR all frames (incl. hits=0)", near_avg_all, near_max
        );
        println!(
            "{:<30} {:>10.1}   (miss only, {}/{} frames re-cast)",
            "NEAR miss frames only", near_avg_miss, near_misses, n
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "SUBCELL (refine_subcells)", sub_avg, sub_max
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "EMIT (tick + splat)", emit_avg, emit_max
        );
        println!(
            "{:<30} {:>10.1} {:>10.1}",
            "RESOLVE (fold layers)", res_avg, res_max
        );
        println!("{}", "-".repeat(52));
        println!("{:<30} {:>10.1} {:>10.1}", "TOTAL", tot_avg, tot_max);
        println!("Implied FPS (1e6/tot_avg):  {fps:.1}");
        println!(
            "Near cache: {hit} HITs / {miss} MISSes / {n} total frames  ({hit_pct:.0}% hit rate)",
            hit = self.near_hits,
            miss = n.saturating_sub(self.near_hits),
            hit_pct = 100.0 * self.near_hits as f64 / n.max(1) as f64,
        );
        println!("Far re-bakes: {}/{n}", self.far_rebakes);
    }
}

// ─── NearKey (mirrors composite.rs) ─────────────────────────────────────────

const NEAR_YAW_BUCKET: f32 = std::f32::consts::TAU / 128.0;
const NEAR_PITCH_BUCKET: f32 = std::f32::consts::TAU / 128.0;

/// Mirrors `composite::ANIM_RECAST_DT` (animation-only near re-cast throttle, ~30 Hz).
const ANIM_RECAST_DT: f32 = 1.0 / 30.0;

/// Simulated per-frame wall-clock step the harness advances its `Res<Time>` stand-in
/// by. The harness runs flat-out (no real 60 Hz pacing), so it advances a SIMULATED
/// clock at a fixed 60 FPS to exercise the framerate-independent throttle exactly as
/// the live system would at 60 Hz: an animation-only re-cast every other frame (~30 Hz).
const SIM_FRAME_DT: f32 = 1.0 / 60.0;

#[derive(Clone, Copy, PartialEq, Eq)]
struct NearKey {
    cell_x: i32,
    cell_y: i32,
    z_bucket: i32,
    yaw_bucket: i32,
    pitch_bucket: i32,
    cols: usize,
    rows: usize,
}

impl NearKey {
    fn of(cam: &CameraPose, cols: usize, rows: usize) -> Self {
        NearKey {
            cell_x: cam.pos.x.floor() as i32,
            cell_y: cam.pos.y.floor() as i32,
            z_bucket: (cam.pos.z * 10.0).round() as i32,
            yaw_bucket: (cam.yaw / NEAR_YAW_BUCKET).round() as i32,
            pitch_bucket: (cam.pitch / NEAR_PITCH_BUCKET).round() as i32,
            cols,
            rows,
        }
    }
}

// ─── Far dirty state (mirrors CompositeDirty) ────────────────────────────────

struct FarDirty {
    bake_frame: u64,
    bake_yaw: f32,
    bake_x: f32,
    bake_y: f32,
    /// Last (yaw, pitch, cols, rows) the far layer was FILLED at — mirrors
    /// `composite::CompositeDirty::far_fill_last`. `None` forces the first fill.
    fill_last: Option<(f32, f32, usize, usize)>,
}

impl Default for FarDirty {
    fn default() -> Self {
        FarDirty {
            bake_frame: u64::MAX, // force first bake
            bake_yaw: f32::NAN,
            bake_x: f32::NAN,
            bake_y: f32::NAN,
            fill_last: None,
        }
    }
}

// ─── Single scenario run ──────────────────────────────────────────────────────

/// The shared scene + grid inputs every scenario casts against (bundled so the
/// per-scenario call stays under clippy's argument-count lint).
struct Scene<'a> {
    map: &'a re_world::Map,
    registry: &'a MaterialRegistry,
    fov: &'a Fov,
    emitters: &'a [re_render::Emitter],
    cols: usize,
    rows: usize,
    warmup: usize,
    frames: usize,
}

fn run_scenario(
    scene: &Scene<'_>,
    label: &str,
    spawn: CameraPose,
    // per-frame camera delta (IDLE = (0,0,0,0), MOVING = small increment)
    move_fn: impl Fn(usize, &CameraPose) -> CameraPose,
    // When true, ANIMATION-ONLY near re-casts are throttled to ANIM_RECAST_DT (the
    // live behaviour). When false, animation re-casts EVERY frame (the pre-fix
    // baseline) — used only to print the BEFORE number in the ANIMATED-IDLE scenario.
    throttle_anim: bool,
    // When true, an animation-only frame (low animated fraction) runs the SELECTIVE
    // re-shade (animated cells only, from the cached G-buffer) EVERY frame, mirroring
    // `composite_render_sys`. When false, the legacy throttled-full-re-cast path.
    selective: bool,
) -> Timings {
    let Scene {
        map,
        registry,
        fov,
        emitters,
        cols,
        rows,
        warmup,
        frames,
    } = *scene;
    // Allocate persistent state (mirrors the Bevy resources).
    let mut stack = CompositeStack::new(cols, rows);
    let mut far = FarCache::new(rows);
    let mut scratch = Framebuffer::new(cols, rows);
    let mut fb = Framebuffer::new(cols, rows);
    let mut pool = ParticlePool::default();
    let mut clock = TierClock::default();
    let mut far_dirty = FarDirty::default();
    let mut near_last: Option<NearKey> = None;
    // Simulated wall-clock seconds (stand-in for `Res<Time>::elapsed_secs`) and the
    // time of the last near re-cast — mirrors `CompositeDirty::last_near_recast_secs`.
    let mut sim_secs: f32 = 0.0;
    let mut last_near_recast_secs: f32 = f32::NEG_INFINITY;
    // PERSISTENT sub-cell marks buffer — mirrors `NearScratch.marks` in the live
    // system so the SUBCELL stage pays NO per-frame heap alloc (reused via .clear()).
    let mut subcell_marks: Vec<(usize, usize)> = Vec::new();
    // PERSISTENT selective-re-shade G-buffer — mirrors `NearScratch.gbuf`.
    let mut gbuf = NearGBuffer::default();
    let far_freeze = FarFreeze::default().distance;
    let mut last_anim_frac = 0.0f64;
    let mut selective_count = 0usize;

    let mut cam = spawn;

    let mut timings = Timings::default();
    let total_frames = warmup + frames;

    for fi in 0..total_frames {
        clock.tick();
        sim_secs += SIM_FRAME_DT;
        cam = move_fn(fi, &cam);

        // Size is fixed in the harness, but mirror composite.rs's `resized` so the
        // far-fill / near-dirty predicates are structurally identical.
        let resized = stack.dims() != (cols, rows);

        // ── FAR ── (mirrors composite.rs:563-583 exactly)
        let dyaw = cam.yaw - far_dirty.bake_yaw;
        let dx = cam.pos.x - far_dirty.bake_x;
        let dy = cam.pos.y - far_dirty.bake_y;
        let moved_cells = (dx * dx + dy * dy).sqrt();
        let frames_since = clock.frame.wrapping_sub(far_dirty.bake_frame);
        let far_t = Instant::now();
        let mut far_rebaked = false;
        let force_oneshot = std::env::var("PERFBENCH_ONESHOT").is_ok();
        if force_oneshot {
            // BEFORE baseline: the pre-change behaviour — re-bake the WHOLE ring in
            // one call every frame the tier policy fires (the ~39 ms spike).
            if !far.valid || far_needs_rebake(frames_since, dyaw, moved_cells) {
                far.rebake(map, &cam, registry);
                far_dirty.bake_frame = clock.frame;
                far_dirty.bake_yaw = cam.yaw;
                far_dirty.bake_x = cam.pos.x;
                far_dirty.bake_y = cam.pos.y;
                far_rebaked = true;
            }
        } else if !far.valid {
            // First-ever bake: one shot (mirrors composite.rs).
            far.rebake(map, &cam, registry);
            far_dirty.bake_frame = clock.frame;
            far_dirty.bake_yaw = cam.yaw;
            far_dirty.bake_x = cam.pos.x;
            far_dirty.bake_y = cam.pos.y;
            far_rebaked = true;
        } else {
            // Steady state: INCREMENTAL bake (~one chunk/frame), keep running an
            // in-progress bake to completion (mirrors composite.rs exactly).
            let in_progress = far.bake_in_progress();
            if far_needs_rebake(frames_since, dyaw, moved_cells) || in_progress {
                let should_restart = if in_progress {
                    let (start_yaw, start_pos) = far.bake_start_pose();
                    let sdx = cam.pos.x - start_pos.x;
                    let sdy = cam.pos.y - start_pos.y;
                    far_needs_rebake(0, cam.yaw - start_yaw, (sdx * sdx + sdy * sdy).sqrt())
                } else {
                    false
                };
                if far.bake_step(map, &cam, registry, should_restart) {
                    far_dirty.bake_frame = clock.frame;
                    far_dirty.bake_yaw = far.base_yaw;
                    far_dirty.bake_x = far.base_pos.x;
                    far_dirty.bake_y = far.base_pos.y;
                    far_rebaked = true;
                }
            }
        }
        // Far-fill idle skip: fill only when yaw/pitch/size moved, or a rebake/
        // resize forced it (mirrors composite.rs's `far_fill_dirty`).
        let fill_key = (cam.yaw, cam.pitch, cols, rows);
        let far_fill_dirty =
            resized || far_rebaked || !far.valid || far_dirty.fill_last != Some(fill_key);
        if far_fill_dirty {
            far.fill_far_layer(&cam, &mut stack.far);
            far_dirty.fill_last = Some(fill_key);
        }
        let far_us = far_t.elapsed().as_secs_f64() * 1_000_000.0;

        // ── NEAR (dirty-flag gated) ── (mirrors composite.rs:571-580 exactly,
        // INCLUDING the map-local animation check that the live system uses).
        let key = NearKey::of(&cam, cols, rows);
        // STRUCTURAL dirty (camera/map/resize) re-casts immediately; ANIMATION-ONLY
        // dirty is throttled to ANIM_RECAST_DT (~30 Hz) against the simulated clock.
        let structural_dirty = resized || near_last != Some(key);
        let anim_renderable = re_render::map_local_any_animated(map, registry);
        let grid_cells = (cols * rows).max(1);
        let anim_frac = gbuf.animated_len() as f64 / grid_cells as f64;
        let selective_ok = selective
            && !structural_dirty
            && anim_renderable
            && gbuf.is_valid()
            && anim_frac <= ANIM_SELECTIVE_MAX_FRAC as f64;
        // The throttled-full fallback gate (mirrors composite.rs): used when not
        // selective (legacy/BEFORE) or the animated fraction trips the worst-case guard.
        let throttled_full_due = if throttle_anim {
            !structural_dirty
                && anim_renderable
                && (sim_secs - last_near_recast_secs >= ANIM_RECAST_DT)
        } else {
            // BASELINE: animation re-casts every frame (no throttle).
            !structural_dirty && anim_renderable
        };
        let (near_us, subcell_us);
        if structural_dirty {
            // FULL capturing cast + subcell + skip-flags (mirrors composite.rs).
            let near_t = Instant::now();
            build_world_layers_fov_capture_into(
                map,
                &cam,
                registry,
                fov,
                sim_secs,
                &mut scratch,
                &mut gbuf,
                &mut stack,
            );
            near_us = near_t.elapsed().as_secs_f64() * 1_000_000.0;
            let sub_t = Instant::now();
            refine_subcells_buf(
                map,
                &cam,
                registry,
                fov,
                &mut stack.opaque,
                &mut subcell_marks,
            );
            for &(mc, mr) in subcell_marks.iter() {
                gbuf.mark_skip(mc, mr);
            }
            subcell_us = sub_t.elapsed().as_secs_f64() * 1_000_000.0;
            last_anim_frac = gbuf.animated_len() as f64 / grid_cells as f64;
            near_last = Some(key);
            last_near_recast_secs = sim_secs;
        } else if selective_ok {
            // SELECTIVE re-shade: animated cells only, from cached geometry. Every frame.
            let near_t = Instant::now();
            reshade_animated_into(map, registry, far_freeze, &gbuf, sim_secs, &mut stack);
            near_us = near_t.elapsed().as_secs_f64() * 1_000_000.0;
            subcell_us = 0.0;
            selective_count += 1;
        } else if throttled_full_due {
            // THROTTLED full re-cast (worst-case guard / no G-buffer / not selective).
            let near_t = Instant::now();
            build_world_layers_fov_capture_into(
                map,
                &cam,
                registry,
                fov,
                sim_secs,
                &mut scratch,
                &mut gbuf,
                &mut stack,
            );
            near_us = near_t.elapsed().as_secs_f64() * 1_000_000.0;
            let sub_t = Instant::now();
            refine_subcells_buf(
                map,
                &cam,
                registry,
                fov,
                &mut stack.opaque,
                &mut subcell_marks,
            );
            for &(mc, mr) in subcell_marks.iter() {
                gbuf.mark_skip(mc, mr);
            }
            subcell_us = sub_t.elapsed().as_secs_f64() * 1_000_000.0;
            near_last = Some(key);
            last_near_recast_secs = sim_secs;
        } else {
            near_us = 0.0;
            subcell_us = 0.0;
        }

        // ── EMIT ──
        let emit_t = Instant::now();
        let mut rng = ParticleRng::seed(0xD0E5_1A77 ^ clock.frame);
        tick_emitters(emitters, &mut pool, 1.0 / 60.0, &mut rng);
        let (opaque, emit_layer) = (&stack.opaque, &mut stack.emitters);
        splat_particles(&pool, &cam, fov, opaque, emit_layer);
        let emit_us = emit_t.elapsed().as_secs_f64() * 1_000_000.0;

        // ── RESOLVE ──
        let resolve_t = Instant::now();
        fb.clear();
        stack.resolve(&mut fb);
        let resolve_us = resolve_t.elapsed().as_secs_f64() * 1_000_000.0;

        if fi >= warmup {
            timings.push(
                far_us,
                near_us,
                subcell_us,
                emit_us,
                resolve_us,
                far_rebaked,
            );
        }
    }

    timings.anim_frac = last_anim_frac;
    timings.selective_frames = selective_count;
    timings.print_table(label, frames);
    timings
}

/// As [`run_scenario`] but prints the extended motion table (near-miss breakdown).
fn run_motion_scenario(
    scene: &Scene<'_>,
    label: &str,
    spawn: CameraPose,
    move_fn: impl Fn(usize, &CameraPose) -> CameraPose,
) -> Timings {
    let Scene {
        map,
        registry,
        fov,
        emitters,
        cols,
        rows,
        warmup,
        frames,
    } = *scene;
    let mut stack = CompositeStack::new(cols, rows);
    let mut far = FarCache::new(rows);
    let mut scratch = Framebuffer::new(cols, rows);
    let mut fb = Framebuffer::new(cols, rows);
    let mut pool = ParticlePool::default();
    let mut clock = TierClock::default();
    let mut far_dirty = FarDirty::default();
    let mut near_last: Option<NearKey> = None;
    let mut sim_secs: f32 = 0.0;
    let mut last_near_recast_secs: f32 = f32::NEG_INFINITY;
    let mut subcell_marks: Vec<(usize, usize)> = Vec::new();
    let mut gbuf = NearGBuffer::default();
    let _far_freeze = FarFreeze::default().distance;
    let mut last_anim_frac = 0.0f64;

    // NEAR rotation-pan ring (mirrors `composite::NearRingRes`). Sized to the live view
    // + FOV; pure rotation within its margin pans it with ZERO near rays.
    let view = view_params(cols, rows, fov);
    let plane_scale = view.plane_scale;
    let mut near_ring = NearRing::new(cols, rows, view.hfov);
    let has_special = re_render::map_local_has_special_cells(map, registry);
    let mut ring_bake_frame: u64 = 0;
    const NEAR_RING_SAFETY_FRAMES: u64 = 60;

    let mut cam = spawn;
    let mut timings = Timings::default();
    let total_frames = warmup + frames;
    // Pan instrumentation: count pure-pan frames (pan served, NO bake-step this frame)
    // and their fill-only near cost — the steady-state rotation win once the ring is hot.
    let mut pan_frames = 0usize;
    let mut pan_fill_only_frames = 0usize;
    let mut pan_fill_only_us_sum = 0.0f64;
    // WORLD-ANCHORED TEXTURE CACHE hit-rate: summed over frames that ran a full
    // capturing near cast (the only frames that touch the cross-frame cache).
    let mut tex_hits_sum = 0u64;
    let mut tex_total_sum = 0u64;
    let mut tex_cast_frames = 0usize;
    // SPECIAL-CELL RECAST (mirror/holo on a panned frame) instrumentation: how many
    // columns the live view shows a special cell in (re-cast fresh) vs panned, summed
    // over panned frames that ran the recast, so we can report the special fraction.
    let mut special_recast_frames = 0usize;
    let mut special_cols_sum = 0u64;
    let mut special_col_buf: Vec<usize> = Vec::new();
    let mut special_recast_us_sum = 0.0f64;

    for fi in 0..total_frames {
        clock.tick();
        sim_secs += SIM_FRAME_DT;
        cam = move_fn(fi, &cam);

        let resized = stack.dims() != (cols, rows);

        // ── FAR ──
        let dyaw = cam.yaw - far_dirty.bake_yaw;
        let dx = cam.pos.x - far_dirty.bake_x;
        let dy = cam.pos.y - far_dirty.bake_y;
        let moved_cells = (dx * dx + dy * dy).sqrt();
        let frames_since = clock.frame.wrapping_sub(far_dirty.bake_frame);
        let far_t = Instant::now();
        let mut far_rebaked = false;
        let force_oneshot = std::env::var("PERFBENCH_ONESHOT").is_ok();
        if force_oneshot {
            if !far.valid || far_needs_rebake(frames_since, dyaw, moved_cells) {
                far.rebake(map, &cam, registry);
                far_dirty.bake_frame = clock.frame;
                far_dirty.bake_yaw = cam.yaw;
                far_dirty.bake_x = cam.pos.x;
                far_dirty.bake_y = cam.pos.y;
                far_rebaked = true;
            }
        } else if !far.valid {
            far.rebake(map, &cam, registry);
            far_dirty.bake_frame = clock.frame;
            far_dirty.bake_yaw = cam.yaw;
            far_dirty.bake_x = cam.pos.x;
            far_dirty.bake_y = cam.pos.y;
            far_rebaked = true;
        } else {
            let in_progress = far.bake_in_progress();
            if far_needs_rebake(frames_since, dyaw, moved_cells) || in_progress {
                let should_restart = if in_progress {
                    let (start_yaw, start_pos) = far.bake_start_pose();
                    let sdx = cam.pos.x - start_pos.x;
                    let sdy = cam.pos.y - start_pos.y;
                    far_needs_rebake(0, cam.yaw - start_yaw, (sdx * sdx + sdy * sdy).sqrt())
                } else {
                    false
                };
                if far.bake_step(map, &cam, registry, should_restart) {
                    far_dirty.bake_frame = clock.frame;
                    far_dirty.bake_yaw = far.base_yaw;
                    far_dirty.bake_x = far.base_pos.x;
                    far_dirty.bake_y = far.base_pos.y;
                    far_rebaked = true;
                }
            }
        }
        let fill_key = (cam.yaw, cam.pitch, cols, rows);
        let far_fill_dirty =
            resized || far_rebaked || !far.valid || far_dirty.fill_last != Some(fill_key);
        if far_fill_dirty {
            far.fill_far_layer(&cam, &mut stack.far);
            far_dirty.fill_last = Some(fill_key);
        }
        let far_us = far_t.elapsed().as_secs_f64() * 1_000_000.0;

        // ── NEAR ──
        let key = NearKey::of(&cam, cols, rows);
        let structural_dirty = resized || near_last != Some(key);
        let anim_renderable = re_render::map_local_any_animated(map, registry);
        let grid_cells = (cols * rows).max(1);
        // Motion scenarios: no selective (all structural misses on rotation/translation).
        // Use the throttled-full path for any remaining animation-only frames.
        let throttled_full_due = !structural_dirty
            && anim_renderable
            && (sim_secs - last_near_recast_secs >= ANIM_RECAST_DT);

        // ROTATION-PAN (mirrors composite_render_sys): a pure rotation within the ring's
        // margin (and a position move below the parallax threshold), with no no-pan
        // special cells, PANS the baked ring with zero near rays.
        let pan_eligible = !resized && near_ring.can_pan(&cam, plane_scale);
        let (near_us, subcell_us);
        if pan_eligible {
            let near_t = Instant::now();
            let in_progress = near_ring.bake_in_progress();
            let ring_safety = clock.frame.wrapping_sub(ring_bake_frame) >= NEAR_RING_SAFETY_FRAMES;
            let dyaw_edge = re_render::near_cache::ang_delta_pub(cam.yaw, near_ring.base_yaw).abs();
            let drift = dyaw_edge >= near_ring.margin() * 0.5
                || (cam.pos.x - near_ring.base_pos.x).abs() >= 0.25
                || (cam.pos.y - near_ring.base_pos.y).abs() >= 0.25;
            let baked_this_frame = in_progress || drift || ring_safety;
            if baked_this_frame {
                let should_restart = if in_progress {
                    let (_sy, sp) = near_ring.bake_start_pose();
                    (cam.pos.x - sp.x).abs() >= 0.5 || (cam.pos.y - sp.y).abs() >= 0.5
                } else {
                    false
                };
                if near_ring.bake_step(map, &cam, registry, fov, sim_secs, should_restart) {
                    ring_bake_frame = clock.frame;
                }
            }
            near_ring.fill_near_layers(
                &cam,
                plane_scale,
                &mut stack.opaque,
                &mut stack.transparent,
            );
            // SPECIAL-CELL RECAST: pan the static geometry, re-cast only the mirror/holo
            // columns at the live view + overwrite them in the panned layers (EXACT).
            if has_special {
                if scratch.cols() != cols || scratch.rows() != rows {
                    scratch.resize(cols, rows);
                }
                scratch.clear();
                re_render::recast_special_columns_into(
                    map,
                    &cam,
                    registry,
                    fov,
                    sim_secs,
                    &mut scratch,
                    &mut special_col_buf,
                );
                for &col in special_col_buf.iter() {
                    for row in 0..rows {
                        let Some(cell) = scratch.get(col, row).copied() else {
                            continue;
                        };
                        let mat_id = re_render::surface_material(cell.surface, cell.material);
                        if registry.get(mat_id).is_transparent() {
                            stack.transparent.set(col, row, cell);
                        } else {
                            stack.opaque.set(col, row, cell);
                        }
                    }
                }
                if fi >= warmup {
                    special_recast_frames += 1;
                    special_cols_sum += special_col_buf.len() as u64;
                }
            }
            near_us = near_t.elapsed().as_secs_f64() * 1_000_000.0;
            subcell_us = 0.0;
            if fi >= warmup {
                pan_frames += 1;
                if has_special {
                    special_recast_us_sum += near_us;
                }
                if !baked_this_frame {
                    pan_fill_only_frames += 1;
                    pan_fill_only_us_sum += near_us;
                }
            }
        } else if structural_dirty || throttled_full_due {
            let near_t = Instant::now();
            build_world_layers_fov_capture_into(
                map,
                &cam,
                registry,
                fov,
                sim_secs,
                &mut scratch,
                &mut gbuf,
                &mut stack,
            );
            near_us = near_t.elapsed().as_secs_f64() * 1_000_000.0;
            // WORLD-ANCHORED TEXTURE CACHE hit-rate for this near cast (cross-frame
            // reuse of the expensive fBm/relief texture half).
            let (th, tm) = re_render::last_tex_cache_stats();
            if fi >= warmup && th + tm > 0 {
                tex_hits_sum += th;
                tex_total_sum += th + tm;
                tex_cast_frames += 1;
            }
            let sub_t = Instant::now();
            refine_subcells_buf(
                map,
                &cam,
                registry,
                fov,
                &mut stack.opaque,
                &mut subcell_marks,
            );
            for &(mc, mr) in subcell_marks.iter() {
                gbuf.mark_skip(mc, mr);
            }
            subcell_us = sub_t.elapsed().as_secs_f64() * 1_000_000.0;
            last_anim_frac = gbuf.animated_len() as f64 / grid_cells as f64;
            near_last = Some(key);
            last_near_recast_secs = sim_secs;
            // Prime the ring at the new pose so the next rotation can pan (one chunk per
            // structural frame; restart on a move). Primed EVEN with special cells: the
            // ring bakes the static geometry and the next rotation re-casts only the
            // special columns over the pan (see the pan branch).
            {
                let in_progress = near_ring.bake_in_progress();
                let should_restart = if in_progress {
                    let (_sy, sp) = near_ring.bake_start_pose();
                    (cam.pos.x - sp.x).abs() >= 0.5 || (cam.pos.y - sp.y).abs() >= 0.5
                } else {
                    !near_ring.can_pan(&cam, plane_scale)
                };
                if (in_progress || should_restart)
                    && near_ring.bake_step(map, &cam, registry, fov, sim_secs, should_restart)
                {
                    ring_bake_frame = clock.frame;
                }
            }
        } else {
            near_us = 0.0;
            subcell_us = 0.0;
        }

        // ── EMIT ──
        let emit_t = Instant::now();
        let mut rng = ParticleRng::seed(0xD0E5_1A77 ^ clock.frame);
        tick_emitters(emitters, &mut pool, 1.0 / 60.0, &mut rng);
        let (opaque, emit_layer) = (&stack.opaque, &mut stack.emitters);
        splat_particles(&pool, &cam, fov, opaque, emit_layer);
        let emit_us = emit_t.elapsed().as_secs_f64() * 1_000_000.0;

        // ── RESOLVE ──
        let resolve_t = Instant::now();
        fb.clear();
        stack.resolve(&mut fb);
        let resolve_us = resolve_t.elapsed().as_secs_f64() * 1_000_000.0;

        if fi >= warmup {
            timings.push(
                far_us,
                near_us,
                subcell_us,
                emit_us,
                resolve_us,
                far_rebaked,
            );
        }
    }

    timings.anim_frac = last_anim_frac;
    timings.print_motion_table(label, frames);
    if pan_frames > 0 {
        let fill_avg = if pan_fill_only_frames > 0 {
            pan_fill_only_us_sum / pan_fill_only_frames as f64
        } else {
            0.0
        };
        println!(
            "  NEAR ROTATION-PAN: {pan_frames}/{frames} frames panned the ring \
             ({pan_fill_only_frames} pure-pan, no bake) | pure-pan near avg: {fill_avg:.1} µs \
             (the steady-state rotation win; bake frames amortise the rebake)"
        );
    } else {
        println!("  NEAR ROTATION-PAN: ring did not engage this scenario (translation-dominated).");
    }
    if has_special && special_recast_frames > 0 {
        let grid_cols = cols as u64;
        let avg_special = special_cols_sum as f64 / special_recast_frames as f64;
        let frac = 100.0 * special_cols_sum as f64
            / (special_recast_frames as u64 * grid_cols).max(1) as f64;
        let recast_avg = special_recast_us_sum / special_recast_frames as f64;
        println!(
            "  SPECIAL-CELL RECAST: {special_recast_frames} panned frames re-cast mirror/holo \
             columns | avg {avg_special:.1}/{grid_cols} cols special ({frac:.1}% re-cast, \
             {:.1}% panned) | panned-near avg (fill + special recast): {recast_avg:.1} µs",
            100.0 - frac
        );
    } else if has_special {
        println!(
            "  SPECIAL-CELL RECAST: scene has special cells but no panned frame ran the recast \
             (ring did not engage; full casts every frame)."
        );
    }
    if tex_cast_frames > 0 {
        let rate = 100.0 * tex_hits_sum as f64 / tex_total_sum.max(1) as f64;
        println!(
            "  WORLD-TEX CACHE: {tex_cast_frames} full-cast frames | {tex_hits_sum}/{tex_total_sum} \
             texture lookups served from cache => {rate:.1}% cross-frame hit-rate \
             (fBm/relief recompute avoided; lighting still re-run fresh)"
        );
    } else {
        println!("  WORLD-TEX CACHE: no full near casts touched the cross-frame cache this run.");
    }
    timings
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    println!("=================================================================");
    println!("re_render perfbench");
    println!("  level:          {}", cfg.level);
    println!("  complex-level:  {}", cfg.complex_level);
    println!("  classic grid:   {}x{}", cfg.cols, cfg.rows);
    println!(
        "  scenario grid:  {}x{}",
        cfg.scenario_cols, cfg.scenario_rows
    );
    println!("  frames:         {} (+ {} warmup)", cfg.frames, cfg.warmup);
    println!("=================================================================");

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO BW — Big-world procedural brickmap baseline
    //
    // Selected by the positional `bigworld` arg. Runs ONLY this scenario so the
    // numbers it prints are the clean octree-empty-space-skipping baseline.
    //
    // Builds a 512x512x64 sparse procedural world via Map::generate_brickmap,
    // places the camera near the centre at a low (dense) z, and
    // times N frames of the EXISTING uniform-DDA composite cast at the bench grid
    // via run_scenario (the same path scenarios A-E use, IDLE / camera fixed).
    //
    // HONEST CAVEAT: this big world is mostly air at higher z. A ray that crosses a
    // long stretch of empty cells before reaching a wall can hit the uniform DDA's
    // MAX_DDA_STEPS (256) cap WITHOUT finding a surface — that capped walk through
    // empty space IS the cost an octree empty-space-skipping pass is meant to remove.
    // The per-frame µs below therefore include that worst-case air march.
    // ═══════════════════════════════════════════════════════════════════════════
    if cfg.bigworld {
        const BW_SX: usize = 512;
        const BW_SY: usize = 512;
        const BW_SZ: usize = 64;
        const BW_SEED: u64 = 7;

        println!();
        println!("=================================================================");
        println!("SCENARIO BW — Big-world procedural brickmap (octree baseline)");
        println!("=================================================================");
        println!(
            "Generating Map::generate_brickmap({BW_SX}, {BW_SY}, {BW_SZ}, {BW_SEED}) \
             ({} cells) ...",
            BW_SX * BW_SY * BW_SZ
        );

        let gen_t = Instant::now();
        let bw_map = re_world::Map::generate_brickmap(BW_SX, BW_SY, BW_SZ, BW_SEED);
        let gen_secs = gen_t.elapsed().as_secs_f64();

        let (sx, sy, sz) = bw_map.size();
        let solids = bw_map.count(re_world::CellKind::WallSolid)
            + bw_map.count(re_world::CellKind::Floor)
            + bw_map.count(re_world::CellKind::WallMatrix);
        let total_cells = sx * sy * sz;
        let fill_pct = 100.0 * solids as f64 / total_cells as f64;

        println!(
            "  world gen time:  {:.3} s  ({:.1} ms)",
            gen_secs,
            gen_secs * 1000.0
        );
        println!("  world size:      {sx} x {sy} x {sz}  = {total_cells} cells");
        println!("  solid cells:     {solids}  ({fill_pct:.1}% of grid; rest is sparse air)");

        let bw_registry = registry_for_map(&bw_map);
        let fov = Fov::default();
        let emitters = demo_emitters();

        // Camera near the centre at a low z (the procedural threshold makes low z
        // dense ground, high z air — see re_world::gen). spawn_at centres the camera
        // in its cell; facing +y. z=2 keeps it in the dense band.
        let bw_spawn = CameraPose::spawn_at((BW_SX / 2) as i32, (BW_SY / 2) as i32, 2, 0);
        println!(
            "  camera spawn:    centre ({}, {}, 2)  facing +y",
            BW_SX / 2,
            BW_SY / 2
        );
        println!(
            "  bench grid:      {}x{} = {} cells",
            cfg.cols,
            cfg.rows,
            cfg.cols * cfg.rows
        );
        println!(
            "  uniform-DDA cap: MAX_DDA_STEPS=256 — air rays that cross >256 empty\n\
             \x20                  cells before a wall hit the cap (the empty-space-skip\n\
             \x20                  cost an octree pass will remove); included in the µs below."
        );

        let bw_scene = Scene {
            map: &bw_map,
            registry: &bw_registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.cols,
            rows: cfg.rows,
            warmup: cfg.warmup,
            frames: cfg.frames,
        };

        // IDLE (camera fixed): the per-frame composite cast over the big brickmap.
        // The NEAR stage's first cast is the full uniform-DDA cast over this world —
        // that is the baseline measurement. (Same run_scenario path as scenarios A-E.)
        let bw_timings = run_scenario(
            &bw_scene,
            "SCENARIO BW — BIG-WORLD IDLE (512x512x64 procedural brickmap, uniform DDA)",
            bw_spawn,
            |_fi, cam| *cam,
            true,
            false,
        );

        // Headline. IDLE caches the near tier after the first cast, so the AVERAGE
        // total is diluted by cache hits and is NOT the figure an octree pass must
        // beat. The baseline is the FULL UNIFORM-DDA CAST cost: the near-MISS frames
        // (every frame that actually re-walks the world). Report both, and call the
        // full cast out explicitly as the octree target.
        let (far_avg, _) = Timings::summarize(&bw_timings.far_us);
        let (near_avg_all, near_max) = Timings::summarize(&bw_timings.near_us);
        let (sub_avg_all, _) = Timings::summarize(&bw_timings.subcell_us);
        let (tot_avg, tot_max) = Timings::summarize(&bw_timings.total_us);
        // Near full-cast cost = miss frames only (near_us > 0).
        let near_miss: Vec<f64> = bw_timings
            .near_us
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();
        let near_misses = near_miss.len();
        let (near_cast_avg, _) = Timings::summarize(&near_miss);
        let fps_avg = if tot_avg > 0.0 {
            1_000_000.0 / tot_avg
        } else {
            f64::INFINITY
        };
        let fps_worst = if tot_max > 0.0 {
            1_000_000.0 / tot_max
        } else {
            f64::INFINITY
        };
        let fps_cast = if near_cast_avg > 0.0 {
            1_000_000.0 / (far_avg + near_cast_avg + sub_avg_all)
        } else {
            f64::INFINITY
        };

        println!();
        println!("=================================================================");
        println!("BIG-WORLD BASELINE (an octree pass must beat this) — {sx}x{sy}x{sz} brickmap");
        println!(
            "  world gen:               {:.1} ms (one-time)",
            gen_secs * 1000.0
        );
        println!(
            "  FULL UNIFORM-DDA CAST:   {near_cast_avg:>9.1} µs near (avg over {near_misses}/{} miss frames)",
            bw_timings.n
        );
        println!("    near max (worst air march): {near_max:>9.1} µs");
        println!(
            "    cast frame (FAR+NEAR+SUBCELL): {:>7.1} µs  => {fps_cast:.1} FPS  <== OCTREE TARGET",
            far_avg + near_cast_avg + sub_avg_all
        );
        println!("  near avg (all frames, incl. cache hits): {near_avg_all:>7.1} µs");
        println!(
            "  total avg (full frame, cache-diluted):   {tot_avg:>7.1} µs => {fps_avg:.1} FPS"
        );
        println!(
            "  total max (worst / full-cast frame):     {tot_max:>7.1} µs => {fps_worst:.1} FPS"
        );
        println!();
        println!("  NOTE: the uniform DDA walks empty air up to MAX_DDA_STEPS=256 per ray");
        println!("        before giving up — that capped empty-space march is exactly the");
        println!("        cost an octree empty-space-skipping pass is intended to eliminate.");
        println!("=================================================================");

        return;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CLASSIC SCENARIOS A-E — level 01, original 240x64 grid (kept unchanged)
    // ═══════════════════════════════════════════════════════════════════════════
    if !cfg.skip_classic {
        // Build the map + registry exactly as the game does.
        let (map, spawn_cam) = match re_level::load(&cfg.level) {
            Ok(lvl) => {
                let cam = lvl.spawn_camera();
                let m = lvl.build_map();
                (m, cam)
            }
            Err(e) => {
                eprintln!("Could not load level '{}': {e}", cfg.level);
                eprintln!("Falling back to built-in test room.");
                let m = re_world::test_room();
                let cam = re_world::CameraPose::spawn_in_test_room();
                (m, cam)
            }
        };
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let emitters = demo_emitters();

        println!(
            "\nMap size: {}x{}x{}",
            map.size().0,
            map.size().1,
            map.size().2
        );
        println!(
            "Grid cells: {}  ({}x{})",
            cfg.cols * cfg.rows,
            cfg.cols,
            cfg.rows
        );
        println!("Demo emitters: {}", emitters.len());

        let scene = Scene {
            map: &map,
            registry: &registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.cols,
            rows: cfg.rows,
            warmup: cfg.warmup,
            frames: cfg.frames,
        };

        // ── SCENARIO A: IDLE ──────────────────────────────────────────────────────
        let idle_timings = run_scenario(
            &scene,
            "SCENARIO A — IDLE (camera fixed)",
            spawn_cam,
            |_fi, cam| *cam,
            true,
            false,
        );

        // ── SCENARIO B: MOVING ────────────────────────────────────────────────────
        let yaw_step = NEAR_YAW_BUCKET * 1.5_f32;
        let _moving_timings = run_scenario(
            &scene,
            "SCENARIO B — MOVING (forward creep + yaw turn, worst-case cache miss)",
            spawn_cam,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, 0.05, 0.0),
                yaw: cam.yaw + yaw_step,
                pitch: cam.pitch,
            },
            true,
            false,
        );

        // ── SCENARIO C: IDLE with a STATIC-SURFACE registry ───────────────────────
        let static_registry = static_surface_registry();
        let static_local_animated = re_render::map_local_any_animated(&map, &static_registry);
        let static_scene = Scene {
            map: &map,
            registry: &static_registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.cols,
            rows: cfg.rows,
            warmup: cfg.warmup,
            frames: cfg.frames,
        };
        let static_idle = run_scenario(
            &static_scene,
            "SCENARIO C — IDLE, STATIC-SURFACE registry (near cache CAN engage)",
            spawn_cam,
            |_fi, cam| *cam,
            true,
            false,
        );
        println!("\n(static-surface registry: map_local_any_animated = {static_local_animated})");

        // ── IDLE cache validation ─────────────────────────────────────────────────
        println!();
        println!("=================================================================");
        println!("IDLE CACHE VALIDATION");
        println!(
            "NOTE: level 01's circuit/matrix walls are authored-animated, so its near\n      \
             tier legitimately re-casts on idle. SCENARIO C (static surfaces) shows the\n      \
             near-cache idle win the fix unlocks: {}/{} idle frames cached.",
            static_idle.near_hits, cfg.frames
        );
        let hit_pct = 100.0 * idle_timings.near_hits as f64 / cfg.frames as f64;
        if idle_timings.near_hits == cfg.frames {
            println!(
                "  PASS — all {}/{} idle frames hit the near cache (0 near rays).",
                idle_timings.near_hits, cfg.frames
            );
        } else {
            println!(
                "  WARN — only {}/{} idle frames ({:.0}%) hit the near cache.",
                idle_timings.near_hits, cfg.frames, hit_pct
            );
            println!("         Expected 100% after warmup (1 initial cast then ~0 rays).");
        }
        let idle_far_rebakes = idle_timings.far_rebakes;
        println!("  Far re-bakes during idle (post-warmup): {idle_far_rebakes}");
        if idle_far_rebakes == 0 {
            println!("  PASS — far panorama served from cache all idle frames.");
        } else {
            println!("  NOTE — {idle_far_rebakes} safety-tick re-bake(s) within the idle window (expected ≤1).");
        }
        println!("=================================================================");

        // ── SCENARIO D: ANIMATED-IDLE (the throttle win) ──────────────────────────
        let anim_frames = 120usize;
        let anim_scene = Scene {
            map: &map,
            registry: &registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.cols,
            rows: cfg.rows,
            warmup: cfg.warmup,
            frames: anim_frames,
        };
        let anim_before = run_scenario(
            &anim_scene,
            "SCENARIO D — ANIMATED-IDLE, throttle-only (BEFORE: ~30 Hz full re-cast)",
            spawn_cam,
            |_fi, cam| *cam,
            true,
            false,
        );
        let anim_after = run_scenario(
            &anim_scene,
            "SCENARIO E — ANIMATED-IDLE, SELECTIVE re-shade (AFTER: animated cells only)",
            spawn_cam,
            |_fi, cam| *cam,
            true,
            true,
        );

        let (before_avg, _) = Timings::summarize(&anim_before.near_us);
        let (after_avg, _) = Timings::summarize(&anim_after.near_us);
        let speedup = if after_avg > 0.0 {
            before_avg / after_avg
        } else {
            f64::INFINITY
        };
        let moving_check = run_scenario(
            &scene,
            "SCENARIO B' — MOVING (selective ON: must be UNCHANGED — full re-cast every frame)",
            spawn_cam,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, 0.05, 0.0),
                yaw: cam.yaw + yaw_step,
                pitch: cam.pitch,
            },
            true,
            true,
        );

        println!();
        println!("=================================================================");
        println!("ANIMATED-IDLE SELECTIVE-RE-SHADE RESULT (level 01, camera fixed, anim on)");
        println!(
            "  Animated-cell fraction (this view): {:.1}%  (selective engages when ≤ {:.0}%)",
            anim_after.anim_frac * 100.0,
            ANIM_SELECTIVE_MAX_FRAC * 100.0
        );
        println!(
            "  BEFORE (throttle-only, ~30 Hz full re-cast):  avg near {before_avg:>8.1} µs/frame  \
                  ({}/{} frames re-cast)",
            anim_frames - anim_before.near_hits,
            anim_frames
        );
        println!(
            "  AFTER  (selective re-shade, every frame):     avg near {after_avg:>8.1} µs/frame  \
                  ({} selective frames)",
            anim_after.selective_frames
        );
        println!("  Speedup (before/after avg near): {speedup:.2}x");
        if speedup >= 1.6 {
            println!("  PASS — animation-only near cost cut substantially by selective re-shade.");
        } else {
            println!("  NOTE — speedup below ~1.6x; check the animated fraction for this view.");
        }
        println!(
            "  MOVING unchanged check: {} selective frames, {}/{} near hits (both must be 0).",
            moving_check.selective_frames, moving_check.near_hits, cfg.frames
        );
        if moving_check.near_hits == 0 && moving_check.selective_frames == 0 {
            println!("  PASS — MOVING full-re-casts every frame (selective never touches motion).");
        } else {
            println!(
                "  FAIL — MOVING gained selective/cache frames; structural path was bypassed."
            );
        }
        println!("=================================================================");
    } // end !skip_classic

    // ═══════════════════════════════════════════════════════════════════════════
    // NEW SCENARIOS F/G/H — rotation/translation/mixed at 280x72 on complex level
    // ═══════════════════════════════════════════════════════════════════════════
    if !cfg.skip_motion {
        // Yaw sweep rate for PURE_ROTATION: 0.03 rad/frame.
        // Bucket width = TAU/128 ≈ 0.04909 rad.
        // Bucket crossing interval = 0.04909 / 0.03 ≈ 1.64 frames → near re-cast
        // almost every frame. CONFIRM: near_hits ≈ 0; near µs ≈ full cast cost.
        const PURE_ROT_YAW_STEP: f32 = 0.03;
        // Small pitch oscillation (±3 deg at 0.01 rad/frame) to simulate realistic
        // mouse-look. Pitch bucket = TAU/128 ≈ 0.049 rad, so 0.01 rad/frame stays
        // within one bucket most of the time (pitch rarely crosses a bucket).
        const PURE_ROT_PITCH_AMP: f32 = 0.05; // radians total swing
        const PURE_ROT_PITCH_FREQ: f32 = 0.02; // radians/frame (very slow oscillation)

        // Forward step for TRANSLATION: ~0.1 cell/frame in the +y direction.
        // Cell boundary every 10 frames → 12 near re-casts over 120 frames.
        const TRANS_STEP: f32 = 0.1;

        // MIXED: smaller walk + smaller rotation, both each frame.
        const MIXED_TRANS_STEP: f32 = 0.05;
        const MIXED_YAW_STEP: f32 = 0.02;

        // Motion scenarios use 120 measured frames after warmup.
        let motion_frames = 120usize;

        // ── Load COMPLEX LEVEL ────────────────────────────────────────────────────
        println!();
        println!("=================================================================");
        println!("MOTION SCENARIOS (F/G/H) — complex-walls level baseline");
        println!(
            "Grid: {}x{} = {} cells",
            cfg.scenario_cols,
            cfg.scenario_rows,
            cfg.scenario_cols * cfg.scenario_rows
        );
        println!("=================================================================");

        let (complex_map, complex_spawn) = match re_level::load(&cfg.complex_level) {
            Ok(lvl) => {
                let cam = lvl.spawn_camera();
                let m = lvl.build_map();
                (m, cam)
            }
            Err(e) => {
                eprintln!("Could not load complex level '{}': {e}", cfg.complex_level);
                eprintln!("Falling back to built-in test room.");
                let m = re_world::test_room();
                let cam = re_world::CameraPose::spawn_in_test_room();
                (m, cam)
            }
        };
        let complex_registry = registry_for_map(&complex_map);
        let fov = Fov::default();
        let emitters = demo_emitters();

        println!(
            "\nComplex map size: {}x{}x{}",
            complex_map.size().0,
            complex_map.size().1,
            complex_map.size().2
        );
        println!("Emitters: {}", emitters.len());

        // Print the NearKey bucket width so the user can verify the per-frame
        // re-cast expectation for PURE_ROTATION.
        let bucket_deg = NEAR_YAW_BUCKET.to_degrees();
        let frames_per_bucket = NEAR_YAW_BUCKET / PURE_ROT_YAW_STEP;
        println!();
        println!(
            "NearKey bucket width: {:.4} rad ({:.2} deg)",
            NEAR_YAW_BUCKET, bucket_deg
        );
        println!(
            "PURE_ROTATION yaw step: {:.4} rad/frame => bucket crossing every {:.2} frames",
            PURE_ROT_YAW_STEP, frames_per_bucket
        );
        println!(
            "=> PURE_ROTATION expected near re-cast rate: ~every {:.1} frames ({:.0}% miss rate)",
            frames_per_bucket,
            100.0 * (1.0 - (frames_per_bucket - 1.0).max(0.0) / frames_per_bucket)
        );

        let complex_scene = Scene {
            map: &complex_map,
            registry: &complex_registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.scenario_cols,
            rows: cfg.scenario_rows,
            warmup: cfg.warmup,
            frames: motion_frames,
        };

        // ── SCENARIO F: PURE_ROTATION ─────────────────────────────────────────────
        // Camera POSITION fixed; yaw sweeps continuously at 0.03 rad/frame.
        // This is THE mouse-look case. Currently re-casts the near tier ~every frame
        // (no panoramic cache). near_hits should be ~0; near µs should equal a full
        // cast. This is the exact baseline the panoramic NEAR cache will eliminate.
        println!();
        println!("--- SCENARIO F: PURE_ROTATION (position fixed, yaw +0.03 rad/frame) ---");
        println!("    This is the pre-panorama baseline: near re-casts every ~1.6 frames.");
        println!("    After panoramic cache: near µs should drop to ~0 (window pan only).");
        let rotation_timings = run_motion_scenario(
            &complex_scene,
            "SCENARIO F — PURE_ROTATION (yaw sweep, position fixed) [COMPLEX LEVEL]",
            complex_spawn,
            move |fi, cam| {
                let pitch_osc = PURE_ROT_PITCH_AMP * (fi as f32 * PURE_ROT_PITCH_FREQ).sin();
                CameraPose {
                    pos: cam.pos,
                    yaw: cam.yaw + PURE_ROT_YAW_STEP,
                    pitch: pitch_osc,
                }
            },
        );

        // ── SCENARIO G: TRANSLATION ────────────────────────────────────────────────
        // Camera walks forward +0.1 cell/frame in y; yaw fixed.
        // Parallax case — panoramic cache can serve most frames via column-shift,
        // but MUST still re-cast on cell-boundary crossings (every ~10 frames).
        println!();
        println!("--- SCENARIO G: TRANSLATION (walk +0.1 cell/frame, yaw fixed) ---");
        println!("    Cell boundary every ~10 frames => ~12 re-casts over 120 frames.");
        println!("    Panoramic cache: will still re-cast on cell crossings (parallax).");
        let translation_timings = run_motion_scenario(
            &complex_scene,
            "SCENARIO G — TRANSLATION (forward walk, yaw fixed) [COMPLEX LEVEL]",
            complex_spawn,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, TRANS_STEP, 0.0),
                yaw: cam.yaw,
                pitch: cam.pitch,
            },
        );

        // ── SCENARIO H: MIXED ─────────────────────────────────────────────────────
        // Walk + rotate simultaneously (realistic gameplay: strafe + look).
        // Both position and yaw change every frame → near re-cast almost every frame.
        println!();
        println!("--- SCENARIO H: MIXED (walk 0.05 cell/frame + yaw +0.02 rad/frame) ---");
        println!("    Both position and yaw change every frame.");
        let mixed_timings = run_motion_scenario(
            &complex_scene,
            "SCENARIO H — MIXED (walk + rotate) [COMPLEX LEVEL]",
            complex_spawn,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, MIXED_TRANS_STEP, 0.0),
                yaw: cam.yaw + MIXED_YAW_STEP,
                pitch: cam.pitch,
            },
        );

        // ── Also run level 05 at the same grid for comparison ─────────────────────
        println!();
        println!("=================================================================");
        println!("MOTION SCENARIOS F/G/H — level 05 comparison (same grid)");
        println!("=================================================================");

        let (map05, spawn05) = match re_level::load("levels/05.lvl") {
            Ok(lvl) => {
                let cam = lvl.spawn_camera();
                let m = lvl.build_map();
                (m, cam)
            }
            Err(e) => {
                eprintln!("Could not load levels/05.lvl: {e}");
                let m = re_world::test_room();
                let cam = re_world::CameraPose::spawn_in_test_room();
                (m, cam)
            }
        };
        let registry05 = registry_for_map(&map05);
        println!(
            "\nLevel 05 map size: {}x{}x{}",
            map05.size().0,
            map05.size().1,
            map05.size().2
        );

        let scene05 = Scene {
            map: &map05,
            registry: &registry05,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.scenario_cols,
            rows: cfg.scenario_rows,
            warmup: cfg.warmup,
            frames: motion_frames,
        };

        let rot05 = run_motion_scenario(
            &scene05,
            "SCENARIO F05 — PURE_ROTATION [LEVEL 05 comparison]",
            spawn05,
            move |fi, cam| {
                let pitch_osc = PURE_ROT_PITCH_AMP * (fi as f32 * PURE_ROT_PITCH_FREQ).sin();
                CameraPose {
                    pos: cam.pos,
                    yaw: cam.yaw + PURE_ROT_YAW_STEP,
                    pitch: pitch_osc,
                }
            },
        );

        let trans05 = run_motion_scenario(
            &scene05,
            "SCENARIO G05 — TRANSLATION [LEVEL 05 comparison]",
            spawn05,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, TRANS_STEP, 0.0),
                yaw: cam.yaw,
                pitch: cam.pitch,
            },
        );

        let mixed05 = run_motion_scenario(
            &scene05,
            "SCENARIO H05 — MIXED [LEVEL 05 comparison]",
            spawn05,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, MIXED_TRANS_STEP, 0.0),
                yaw: cam.yaw + MIXED_YAW_STEP,
                pitch: cam.pitch,
            },
        );

        // ── Summary ───────────────────────────────────────────────────────────────
        println!();
        println!("=================================================================");
        println!(
            "MOTION SCENARIO SUMMARY  (grid: {}x{} = {} cells)",
            cfg.scenario_cols,
            cfg.scenario_rows,
            cfg.scenario_cols * cfg.scenario_rows
        );
        println!("=================================================================");
        println!();
        println!("ROTATION/TRANSLATION NEAR-CACHE CONFIRMATION");
        println!("(panoramic cache target: PURE_ROTATION near cost → ~0)");
        println!();

        // Helper: near miss count = frames - near_hits
        let (rot_near_avg, rot_near_max) = Timings::summarize(&rotation_timings.near_us);
        let rot_misses = motion_frames.saturating_sub(rotation_timings.near_hits);
        let rot_miss_samples: Vec<f64> = rotation_timings
            .near_us
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();
        let (rot_miss_avg, _) = Timings::summarize(&rot_miss_samples);
        let (rot_tot_avg, _) = Timings::summarize(&rotation_timings.total_us);
        let rot_fps = if rot_tot_avg > 0.0 {
            1_000_000.0 / rot_tot_avg
        } else {
            f64::INFINITY
        };

        let (trans_near_avg, trans_near_max) = Timings::summarize(&translation_timings.near_us);
        let trans_misses = motion_frames.saturating_sub(translation_timings.near_hits);
        let trans_miss_samples: Vec<f64> = translation_timings
            .near_us
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();
        let (trans_miss_avg, _) = Timings::summarize(&trans_miss_samples);
        let (trans_tot_avg, _) = Timings::summarize(&translation_timings.total_us);
        let trans_fps = if trans_tot_avg > 0.0 {
            1_000_000.0 / trans_tot_avg
        } else {
            f64::INFINITY
        };

        let (mixed_near_avg, mixed_near_max) = Timings::summarize(&mixed_timings.near_us);
        let mixed_misses = motion_frames.saturating_sub(mixed_timings.near_hits);
        let mixed_miss_samples: Vec<f64> = mixed_timings
            .near_us
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();
        let (mixed_miss_avg, _) = Timings::summarize(&mixed_miss_samples);
        let (mixed_tot_avg, _) = Timings::summarize(&mixed_timings.total_us);
        let mixed_fps = if mixed_tot_avg > 0.0 {
            1_000_000.0 / mixed_tot_avg
        } else {
            f64::INFINITY
        };

        println!(
            "COMPLEX LEVEL ({}) at {}x{}:",
            cfg.complex_level, cfg.scenario_cols, cfg.scenario_rows
        );
        println!(
            "{:<12} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}",
            "Scenario", "Misses", "Hit%", "nearAvg µs", "nearMiss µs", "totAvg µs", "FPS"
        );
        println!("{}", "-".repeat(70));
        println!(
            "{:<12} {:>8} {:>8.0}% {:>10.1} {:>10.1} {:>8.1} {:>8.1}",
            "F PURE_ROT",
            rot_misses,
            100.0 * rotation_timings.near_hits as f64 / motion_frames as f64,
            rot_near_avg,
            rot_miss_avg,
            rot_tot_avg,
            rot_fps
        );
        println!(
            "{:<12} {:>8} {:>8.0}% {:>10.1} {:>10.1} {:>8.1} {:>8.1}",
            "G TRANS",
            trans_misses,
            100.0 * translation_timings.near_hits as f64 / motion_frames as f64,
            trans_near_avg,
            trans_miss_avg,
            trans_tot_avg,
            trans_fps
        );
        println!(
            "{:<12} {:>8} {:>8.0}% {:>10.1} {:>10.1} {:>8.1} {:>8.1}",
            "H MIXED",
            mixed_misses,
            100.0 * mixed_timings.near_hits as f64 / motion_frames as f64,
            mixed_near_avg,
            mixed_miss_avg,
            mixed_tot_avg,
            mixed_fps
        );
        println!();

        let (rot05_near_avg, _) = Timings::summarize(&rot05.near_us);
        let rot05_miss_samples: Vec<f64> =
            rot05.near_us.iter().copied().filter(|&v| v > 0.0).collect();
        let (rot05_miss_avg, _) = Timings::summarize(&rot05_miss_samples);
        let (rot05_tot_avg, _) = Timings::summarize(&rot05.total_us);
        let rot05_fps = if rot05_tot_avg > 0.0 {
            1_000_000.0 / rot05_tot_avg
        } else {
            f64::INFINITY
        };
        let rot05_misses = motion_frames.saturating_sub(rot05.near_hits);

        let (trans05_near_avg, _) = Timings::summarize(&trans05.near_us);
        let trans05_miss_samples: Vec<f64> = trans05
            .near_us
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();
        let (trans05_miss_avg, _) = Timings::summarize(&trans05_miss_samples);
        let (trans05_tot_avg, _) = Timings::summarize(&trans05.total_us);
        let trans05_fps = if trans05_tot_avg > 0.0 {
            1_000_000.0 / trans05_tot_avg
        } else {
            f64::INFINITY
        };
        let trans05_misses = motion_frames.saturating_sub(trans05.near_hits);

        let (mixed05_near_avg, _) = Timings::summarize(&mixed05.near_us);
        let mixed05_miss_samples: Vec<f64> = mixed05
            .near_us
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();
        let (mixed05_miss_avg, _) = Timings::summarize(&mixed05_miss_samples);
        let (mixed05_tot_avg, _) = Timings::summarize(&mixed05.total_us);
        let mixed05_fps = if mixed05_tot_avg > 0.0 {
            1_000_000.0 / mixed05_tot_avg
        } else {
            f64::INFINITY
        };
        let mixed05_misses = motion_frames.saturating_sub(mixed05.near_hits);

        println!(
            "LEVEL 05 at {}x{} (comparison):",
            cfg.scenario_cols, cfg.scenario_rows
        );
        println!(
            "{:<12} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}",
            "Scenario", "Misses", "Hit%", "nearAvg µs", "nearMiss µs", "totAvg µs", "FPS"
        );
        println!("{}", "-".repeat(70));
        println!(
            "{:<12} {:>8} {:>8.0}% {:>10.1} {:>10.1} {:>8.1} {:>8.1}",
            "F05 PURE_ROT",
            rot05_misses,
            100.0 * rot05.near_hits as f64 / motion_frames as f64,
            rot05_near_avg,
            rot05_miss_avg,
            rot05_tot_avg,
            rot05_fps
        );
        println!(
            "{:<12} {:>8} {:>8.0}% {:>10.1} {:>10.1} {:>8.1} {:>8.1}",
            "G05 TRANS",
            trans05_misses,
            100.0 * trans05.near_hits as f64 / motion_frames as f64,
            trans05_near_avg,
            trans05_miss_avg,
            trans05_tot_avg,
            trans05_fps
        );
        println!(
            "{:<12} {:>8} {:>8.0}% {:>10.1} {:>10.1} {:>8.1} {:>8.1}",
            "H05 MIXED",
            mixed05_misses,
            100.0 * mixed05.near_hits as f64 / motion_frames as f64,
            mixed05_near_avg,
            mixed05_miss_avg,
            mixed05_tot_avg,
            mixed05_fps
        );

        println!();
        println!("=================================================================");
        println!("NEAR-CACHE STATUS (pre-panoramic-cache baseline)");
        println!("=================================================================");
        println!();
        let rot_miss_pct = 100.0 * rot_misses as f64 / motion_frames as f64;
        let rot_near_max_val = rot_near_max;
        println!("PURE_ROTATION near cache miss rate:  {rot_miss_pct:.0}% ({rot_misses}/{motion_frames} frames re-cast)");
        println!("  near avg (all frames):   {:>8.1} µs", rot_near_avg);
        println!(
            "  near avg (miss frames):  {:>8.1} µs  <- this is the full cast cost",
            rot_miss_avg
        );
        println!("  near max (any frame):    {:>8.1} µs", rot_near_max_val);
        if rot_misses >= (motion_frames * 55 / 100) {
            println!(
                "  CONFIRMED: PURE_ROTATION re-casts near tier every ~{:.1} frames today.",
                frames_per_bucket
            );
            println!("  The panoramic cache will reduce this to 0 near re-casts on pure rotation.");
        } else {
            println!("  NOTE: fewer misses than expected. Check yaw step vs bucket width.");
        }
        println!();
        println!("TRANSLATION near cache miss rate:    {:.0}% ({trans_misses}/{motion_frames} frames re-cast)",
            100.0 * trans_misses as f64 / motion_frames as f64);
        println!("  Expected: ~10% miss rate (cell crossing every ~10 frames at 0.1 cell/frame)");
        println!();
        println!("MIXED near cache miss rate:          {:.0}% ({mixed_misses}/{motion_frames} frames re-cast)",
            100.0 * mixed_misses as f64 / motion_frames as f64);
        println!("=================================================================");

        // Suppress dead-code warnings from the Timings fields we computed but don't
        // print inline; they are all used in the summary table above.
        let _ = (rot_near_max, trans_near_max, mixed_near_max);
    } // end !skip_motion

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO I — Fractal portal material cost (level 92, 240x64)
    // Measures the TextureRecipe::Fractal (portal_zoom / portal_spiral) near cost
    // vs a plain stone wall, and the animated-idle (SpinZoom / Zoom are animated)
    // cost on the portal scene.
    // ═══════════════════════════════════════════════════════════════════════════
    if !cfg.skip_fractal {
        println!();
        println!("=================================================================");
        println!(
            "SCENARIO I — Fractal portal material (level 92, {}x{})",
            cfg.cols, cfg.rows
        );
        println!("=================================================================");

        let (fractal_map, fractal_spawn) = match re_level::load(&cfg.fractal_level) {
            Ok(lvl) => {
                let cam = lvl.spawn_camera();
                let m = lvl.build_map();
                (m, cam)
            }
            Err(e) => {
                eprintln!("Could not load fractal level '{}': {e}", cfg.fractal_level);
                eprintln!("Falling back to built-in test room.");
                let m = re_world::test_room();
                let cam = re_world::CameraPose::spawn_in_test_room();
                (m, cam)
            }
        };
        let fractal_registry = registry_for_map(&fractal_map);
        let fov = Fov::default();
        let emitters = demo_emitters();

        println!(
            "\nFractal portal map size: {}x{}x{}",
            fractal_map.size().0,
            fractal_map.size().1,
            fractal_map.size().2
        );
        println!(
            "Grid cells: {}  ({}x{})",
            cfg.cols * cfg.rows,
            cfg.cols,
            cfg.rows
        );

        // How much of the grid is animated (portal_zoom = Zoom, portal_spiral = SpinZoom).
        let fractal_any_animated =
            re_render::map_local_any_animated(&fractal_map, &fractal_registry);
        println!("map_local_any_animated (fractal portals): {fractal_any_animated}");

        let fractal_scene = Scene {
            map: &fractal_map,
            registry: &fractal_registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.cols,
            rows: cfg.rows,
            warmup: cfg.warmup,
            frames: cfg.frames,
        };

        // I-A: IDLE (camera fixed). With animated portals in view, the near tier
        // throttles re-casts at ~30 Hz (every other simulated frame). We compare
        // this to level 01 (scenario A) to isolate the fractal material overhead.
        println!();
        println!(
            "--- SCENARIO I-A: Fractal portal IDLE (camera fixed, animated portals in view) ---"
        );
        let fractal_idle = run_scenario(
            &fractal_scene,
            "SCENARIO I-A — FRACTAL IDLE (portal_zoom + portal_spiral in view)",
            fractal_spawn,
            |_fi, cam| *cam,
            true,
            true, // selective ON so animated-idle path matches the live system
        );

        // I-B: MOVING (forward creep + yaw, every frame a structural miss).
        // This isolates the per-frame fractal shade cost: every frame re-casts the
        // near tier and shades the portal walls with the Fractal TextureRecipe.
        println!();
        println!("--- SCENARIO I-B: Fractal portal MOVING (worst-case miss per frame) ---");
        let yaw_step_fractal = NEAR_YAW_BUCKET * 1.5_f32;
        let fractal_moving = run_scenario(
            &fractal_scene,
            "SCENARIO I-B — FRACTAL MOVING (forward creep + yaw, fractal walls every frame)",
            fractal_spawn,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, 0.05, 0.0),
                yaw: cam.yaw + yaw_step_fractal,
                pitch: cam.pitch,
            },
            true,
            false,
        );

        // Print the fractal vs plain-wall comparison summary.
        let (fractal_idle_near_avg, fractal_idle_near_max) =
            Timings::summarize(&fractal_idle.near_us);
        let (fractal_moving_near_avg, fractal_moving_near_max) =
            Timings::summarize(&fractal_moving.near_us);
        let (fractal_moving_tot_avg, fractal_moving_tot_max) =
            Timings::summarize(&fractal_moving.total_us);
        let fractal_fps = if fractal_moving_tot_avg > 0.0 {
            1_000_000.0 / fractal_moving_tot_avg
        } else {
            f64::INFINITY
        };

        println!();
        println!("=================================================================");
        println!("FRACTAL PORTAL vs PLAIN WALL — NEAR cost comparison");
        println!(
            "Grid: {}x{} = {} cells",
            cfg.cols,
            cfg.rows,
            cfg.cols * cfg.rows
        );
        println!();
        println!("IDLE (animated portals — throttled re-cast or selective):");
        println!(
            "  Fractal (level 92) near avg: {:>8.1} µs  max {:>8.1} µs  ({} hits/{} frames)",
            fractal_idle_near_avg, fractal_idle_near_max, fractal_idle.near_hits, cfg.frames
        );
        println!("  (level 01 baseline from scenario A/D: see above for comparison)");
        println!();
        println!("MOVING (every frame re-casts near):");
        println!(
            "  Fractal (level 92) near avg: {:>8.1} µs  max {:>8.1} µs",
            fractal_moving_near_avg, fractal_moving_near_max
        );
        println!(
            "  Total avg: {:>8.1} µs  max {:>8.1} µs  => {fractal_fps:.1} FPS",
            fractal_moving_tot_avg, fractal_moving_tot_max
        );
        if fractal_fps >= 60.0 {
            println!("  VERDICT: PASS — fractal portal scene holds >= 60 FPS on MOVING.");
        } else {
            println!("  VERDICT: FAIL — fractal portal scene below 60 FPS on MOVING ({fractal_fps:.1} FPS).");
        }
        println!(
            "  Animated-cell fraction in this view: {:.1}%",
            fractal_idle.anim_frac * 100.0
        );
        println!("=================================================================");

        // ═══════════════════════════════════════════════════════════════════════
        // SCENARIO J — render_tunnel full-fill at two grids
        // Times render_tunnel (fills the WHOLE framebuffer, no geometry) at the
        // classic 240x64 and fullscreen 280x72 grids, reporting per-call µs and
        // the implied single-call FPS so the team knows the ceiling.
        // ═══════════════════════════════════════════════════════════════════════
        println!();
        println!("=================================================================");
        println!("SCENARIO J — render_tunnel full-fill microbench");
        println!("=================================================================");

        let tunnel_warmup = cfg.warmup;
        let tunnel_frames = 300usize; // more frames for stable µs mean

        for &(tc, tr) in &[(cfg.cols, cfg.rows), (cfg.scenario_cols, cfg.scenario_rows)] {
            let mut fb = Framebuffer::new(tc, tr);
            let params = TunnelParams::default(); // arms=5, twist=0.9, octave_warp=0.5
            let mut tunnel_us: Vec<f64> = Vec::with_capacity(tunnel_frames);

            for fi in 0..(tunnel_warmup + tunnel_frames) {
                // Advance t so the animated octave_warp path is exercised (not t=0).
                let t = fi as f32 * (1.0 / 60.0);
                let t_start = Instant::now();
                render_tunnel(&mut fb, t, &params);
                let elapsed = t_start.elapsed().as_secs_f64() * 1_000_000.0;
                if fi >= tunnel_warmup {
                    tunnel_us.push(elapsed);
                }
            }

            let (t_avg, t_max) = Timings::summarize(&tunnel_us);
            let t_fps = if t_avg > 0.0 {
                1_000_000.0 / t_avg
            } else {
                f64::INFINITY
            };
            let cells = tc * tr;

            println!();
            println!("render_tunnel at {tc}x{tr} = {cells} cells:");
            println!(
                "  avg: {:>8.1} µs   max: {:>8.1} µs   implied FPS: {t_fps:.1}",
                t_avg, t_max
            );
            println!("  per-cell: {:.3} µs/cell", t_avg / cells as f64);
            if t_fps >= 60.0 {
                println!("  VERDICT: PASS — tunnel full-fill within a 60 FPS frame budget.");
            } else {
                println!("  VERDICT: FAIL — tunnel full-fill exceeds 60 FPS budget ({t_fps:.1} FPS implied).");
            }
        }
        println!();
        println!("=================================================================");
    } // end !skip_fractal

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO K — Warp-portal GEOMETRY perf gate (levels/93-warp-portal.lvl)
    //
    // This is the ≥60 FPS release gate for the NON-EUCLIDEAN warp-portal render
    // path.
    //
    // What is different from scenarios I/J (fractal material only):
    //
    //   1. The map has REAL PORTAL PAIRS (`Map::portals` is non-empty).
    //      `map_local_has_special_cells` returns TRUE for ANY portal map — the
    //      cache-invalidation rule. This forces the ENTIRE near ring to re-cast fresh
    //      every panned frame — the pan optimisation is bypassed.
    //
    //   2. Each portal-face column pays +1 `dda_hit` (the `rebase_ray` + second
    //      DDA march into the destination region, `cast_column` warp branch).
    //
    // K-A: STATIC (camera fixed, portal opening fully in view).
    //      Measures the forced special-cell recast on a pan frame: PHASE A (primary
    //      DDA + portal-check for ALL columns) + PHASE B (full cast for portal cols).
    //      This is the "idle" cost for a portal scene — it can NEVER be zero while
    //      the portal is in view, unlike a plain-wall scene (no special cells).
    //
    // K-B: MOVING (forward creep + yaw every frame, full structural re-cast).
    //      Measures the combined cost of the forced full re-cast + warp branch on
    //      every portal column in the viewport.
    //
    // K-baseline: same room WITHOUT portals (plain stone walls) for pure overhead
    //      attribution. The cost delta = (forced special recast + warp dda_hit) cost.
    //
    // Grid: 240x64 (the standard classic grid), matching scenarios A-E and I.
    // ═══════════════════════════════════════════════════════════════════════════
    if !cfg.skip_warp {
        println!();
        println!("=================================================================");
        println!("SCENARIO K — Warp-portal GEOMETRY perf gate (≥60 FPS bar)");
        println!("Level: {}", cfg.warp_level);
        println!(
            "Grid: {}x{} = {} cells",
            cfg.cols,
            cfg.rows,
            cfg.cols * cfg.rows
        );
        println!("=================================================================");

        // ── Load the warp-portal level ────────────────────────────────────────
        let (warp_map, warp_spawn) = match re_level::load(&cfg.warp_level) {
            Ok(lvl) => {
                let cam = lvl.spawn_camera();
                let m = lvl.build_map();
                (m, cam)
            }
            Err(e) => {
                eprintln!("Could not load warp level '{}': {e}", cfg.warp_level);
                eprintln!("Falling back to built-in test room.");
                let m = re_world::test_room();
                let cam = re_world::CameraPose::spawn_in_test_room();
                (m, cam)
            }
        };
        let warp_registry = registry_for_map(&warp_map);
        let fov = Fov::default();
        let emitters = demo_emitters();

        println!(
            "\nWarp-portal map size: {}x{}x{}",
            warp_map.size().0,
            warp_map.size().1,
            warp_map.size().2
        );
        println!("Portal pairs in map: {}", warp_map.portals.len());
        let warp_has_special = re_render::map_local_has_special_cells(&warp_map, &warp_registry);
        let warp_any_animated = re_render::map_local_any_animated(&warp_map, &warp_registry);
        println!(
            "map_local_has_special_cells: {warp_has_special}  \
                  (must be TRUE — portals force no-pan near recast)"
        );
        println!(
            "map_local_any_animated:      {warp_any_animated}  \
                  (portal_spiral skin is animated)"
        );

        let warp_scene = Scene {
            map: &warp_map,
            registry: &warp_registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.cols,
            rows: cfg.rows,
            warmup: cfg.warmup,
            frames: cfg.frames,
        };

        // Camera placed facing the north portal opening (portal in full view).
        // Spawn is at level-file [player] spawn: pos≈(6,14,1), facing=-y (north).
        let portal_view_cam = warp_spawn;

        // ── K-A: STATIC (portal in view, camera fixed) ────────────────────────
        // Uses run_motion_scenario so the NearRing + has_special code path fires.
        // With has_special=TRUE: every panned frame runs recast_special_columns_into
        // (Phase A: DDA+portal-check all cols; Phase B: full cast for portal cols).
        // This is the minimum per-frame cost on a portal scene — it cannot be zero.
        println!();
        println!("--- SCENARIO K-A: WARP-PORTAL STATIC (camera fixed, portal in view) ---");
        println!("    Measures the forced special-cell recast: near ring pans the static");
        println!("    geometry, then re-casts portal columns fresh (no cache savings).");
        let warp_static = run_motion_scenario(
            &warp_scene,
            "SCENARIO K-A — WARP-PORTAL STATIC (camera fixed, portal in viewport)",
            portal_view_cam,
            |_fi, cam| *cam,
        );

        // ── K-B: MOVING (forward creep + yaw, structural re-cast every frame) ─
        // Portal columns trigger the warp branch: rebase_ray + second dda_hit per
        // portal column. Camera walks into the portal opening (pos.y decreases from
        // 14 toward 0) so the portal fills more of the viewport over time — worst-case
        // warp column count grows over the run.
        println!();
        println!("--- SCENARIO K-B: WARP-PORTAL MOVING (forward creep + yaw) ---");
        println!("    Measures full structural re-cast + portal warp branch per frame.");
        println!("    Portal columns: +1 dda_hit (rebase_ray + second DDA march).");
        let warp_yaw_step = NEAR_YAW_BUCKET * 1.5_f32;
        let warp_moving = run_motion_scenario(
            &warp_scene,
            "SCENARIO K-B — WARP-PORTAL MOVING (forward walk + yaw, warp branch active)",
            portal_view_cam,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, -0.04, 0.0), // walk toward portal (-y)
                yaw: cam.yaw + warp_yaw_step,
                pitch: cam.pitch,
            },
        );

        // ── K-C: PURE ROTATION (position fixed, yaw sweep) ───────────────────
        // The scenario the near-ring pan optimisation TARGETS. Under view-independent
        // lighting (the default) a portal column's warped destination shade is baked
        // into the near ring at its world angle and PANS exactly like static geometry
        // — so a pure yaw sweep over a portal scene should serve from the ring with
        // zero near re-casts (after the ring is hot), NOT a forced full recast every
        // frame. This isolates the rotation win from the translation parallax of K-B.
        println!();
        println!("--- SCENARIO K-C: WARP-PORTAL PURE ROTATION (position fixed, yaw sweep) ---");
        println!("    Targets the near-ring pan: portal columns bake into the ring (view-");
        println!("    independent warp shade) and PAN — a yaw sweep should serve zero near rays.");
        let warp_rotation = run_motion_scenario(
            &warp_scene,
            "SCENARIO K-C — WARP-PORTAL PURE ROTATION (yaw sweep, position fixed)",
            portal_view_cam,
            move |_fi, cam| CameraPose {
                pos: cam.pos, // FIXED — no translation, no parallax
                yaw: cam.yaw + warp_yaw_step,
                pitch: cam.pitch,
            },
        );

        // ── K-baseline: same camera, plain stone room (NO portals) ───────────
        // Load level 01 (same grid, plain stone walls, no portal pairs) to isolate
        // the warp overhead. We compare NEAR cost here to K-A/K-B to attribute
        // the overhead to: (a) forced full near recast vs (b) portal warp dda_hit.
        println!();
        println!("--- SCENARIO K-baseline: PLAIN STONE MOVING (level 01, no portals) ---");
        println!("    Baseline: same grid, no portals — cost delta vs K-B = warp overhead.");
        let (base_map, base_spawn) = match re_level::load(&cfg.level) {
            Ok(lvl) => (lvl.build_map(), lvl.spawn_camera()),
            Err(_) => (
                re_world::test_room(),
                re_world::CameraPose::spawn_in_test_room(),
            ),
        };
        let base_registry = registry_for_map(&base_map);
        let base_scene = Scene {
            map: &base_map,
            registry: &base_registry,
            fov: &fov,
            emitters: &emitters,
            cols: cfg.cols,
            rows: cfg.rows,
            warmup: cfg.warmup,
            frames: cfg.frames,
        };
        let base_moving = run_motion_scenario(
            &base_scene,
            "SCENARIO K-baseline — MOVING on level 01 (plain stone, no portals)",
            base_spawn,
            move |_fi, cam| CameraPose {
                pos: cam.pos + glam::Vec3::new(0.0, 0.05, 0.0),
                yaw: cam.yaw + warp_yaw_step,
                pitch: cam.pitch,
            },
        );

        // ── Summary and verdict ───────────────────────────────────────────────
        let (warp_static_tot_avg, warp_static_tot_max) = Timings::summarize(&warp_static.total_us);
        let (warp_static_near_avg, warp_static_near_max) = Timings::summarize(&warp_static.near_us);
        let warp_static_fps = if warp_static_tot_avg > 0.0 {
            1_000_000.0 / warp_static_tot_avg
        } else {
            f64::INFINITY
        };

        let (warp_moving_tot_avg, warp_moving_tot_max) = Timings::summarize(&warp_moving.total_us);
        let (warp_moving_near_avg, warp_moving_near_max) = Timings::summarize(&warp_moving.near_us);
        let warp_moving_fps = if warp_moving_tot_avg > 0.0 {
            1_000_000.0 / warp_moving_tot_avg
        } else {
            f64::INFINITY
        };

        let (warp_rot_tot_avg, warp_rot_tot_max) = Timings::summarize(&warp_rotation.total_us);
        let (warp_rot_near_avg, warp_rot_near_max) = Timings::summarize(&warp_rotation.near_us);
        let warp_rot_fps = if warp_rot_tot_avg > 0.0 {
            1_000_000.0 / warp_rot_tot_avg
        } else {
            f64::INFINITY
        };

        let (base_moving_tot_avg, _base_moving_tot_max) = Timings::summarize(&base_moving.total_us);
        let (base_moving_near_avg, _base_moving_near_max) =
            Timings::summarize(&base_moving.near_us);
        let base_moving_fps = if base_moving_tot_avg > 0.0 {
            1_000_000.0 / base_moving_tot_avg
        } else {
            f64::INFINITY
        };

        let warp_overhead_near = warp_moving_near_avg - base_moving_near_avg;
        let warp_overhead_total = warp_moving_tot_avg - base_moving_tot_avg;

        println!();
        println!("=================================================================");
        println!("WARP-PORTAL GEOMETRY PERF GATE — ≥60 FPS on portal scene");
        println!(
            "Grid: {}x{} = {} cells",
            cfg.cols,
            cfg.rows,
            cfg.cols * cfg.rows
        );
        println!();
        println!(
            "{:<40} {:>10} {:>10} {:>8}",
            "Scenario", "near avg µs", "total avg µs", "FPS"
        );
        println!("{}", "-".repeat(72));
        println!(
            "{:<40} {:>10.1} {:>10.1} {:>8.1}",
            "K-A STATIC (portal in view, cam fixed)",
            warp_static_near_avg,
            warp_static_tot_avg,
            warp_static_fps
        );
        println!(
            "{:<40} {:>10.1} {:>10.1} {:>8.1}",
            "K-B MOVING (full re-cast + warp branch)",
            warp_moving_near_avg,
            warp_moving_tot_avg,
            warp_moving_fps
        );
        println!(
            "{:<40} {:>10.1} {:>10.1} {:>8.1}",
            "K-C ROTATION (pan portal ring, pos fixed)",
            warp_rot_near_avg,
            warp_rot_tot_avg,
            warp_rot_fps
        );
        println!(
            "{:<40} {:>10.1} {:>10.1} {:>8.1}",
            "K-baseline MOVING (level 01, no portals)",
            base_moving_near_avg,
            base_moving_tot_avg,
            base_moving_fps
        );
        println!("{}", "-".repeat(72));
        println!(
            "Portal warp NEAR overhead: {:>+.1} µs/frame (K-B minus baseline near)",
            warp_overhead_near
        );
        println!(
            "Portal warp TOTAL overhead: {:>+.1} µs/frame (K-B minus baseline total)",
            warp_overhead_total
        );
        println!();
        println!(
            "Near (pan+special-recast) max: {warp_static_near_max:.1} µs  \
                  total max: {warp_static_tot_max:.1} µs  [K-A STATIC]"
        );
        println!(
            "Near (full re-cast+warp) max:  {warp_moving_near_max:.1} µs  \
                  total max: {warp_moving_tot_max:.1} µs  [K-B MOVING]"
        );
        println!(
            "Near (pan, portal ring) max:   {warp_rot_near_max:.1} µs  \
                  total max: {warp_rot_tot_max:.1} µs  [K-C ROTATION]"
        );
        println!();
        println!("KEY — forced near-ring bypass (map_local_has_special_cells=true):");
        println!("  On a portal map, EVERY panned frame re-casts portal columns fresh.");
        println!(
            "  Static K-A near cost ({:.1} µs) = minimum frame floor for portal scene.",
            warp_static_near_avg
        );
        println!();

        // Dominant cost attribution
        if warp_overhead_near > base_moving_near_avg * 0.5 {
            println!("DOMINANT COST: warp dda_hit overhead (near delta > 50% of baseline near).");
        } else {
            println!("DOMINANT COST: forced full near-recast (overhead < 50% of baseline near).");
            println!(
                "  The forced recast (bypassed pan) is the primary cost, not the warp +1 dda_hit."
            );
        }
        println!();

        // Release-gate verdict: ≥60 FPS on MOVING portal scene
        if warp_moving_fps >= 60.0 {
            println!(
                "VERDICT: PASS — {warp_moving_fps:.1} FPS ≥60 FPS on K-B MOVING (warp geometry)."
            );
        } else {
            println!("VERDICT: FAIL — {warp_moving_fps:.1} FPS <60 FPS on K-B MOVING.");
            println!("  The ≥60 FPS release bar is NOT met for the warp-portal geometry.");
            // Name the dominant stage
            let (warp_far_avg, _) = Timings::summarize(&warp_moving.far_us);
            let (warp_near_avg2, _) = Timings::summarize(&warp_moving.near_us);
            println!("  Frame breakdown: NEAR={warp_near_avg2:.1} µs ({:.0}%), FAR={warp_far_avg:.1} µs ({:.0}%)",
                100.0 * warp_near_avg2 / warp_moving_tot_avg.max(1.0),
                100.0 * warp_far_avg / warp_moving_tot_avg.max(1.0));
        }
        if warp_static_fps >= 60.0 {
            println!("STATIC (idle portal view): PASS — {warp_static_fps:.1} FPS ≥60 FPS.");
        } else {
            println!("STATIC (idle portal view): FAIL — {warp_static_fps:.1} FPS <60 FPS.");
            println!("  Even a STATIC portal scene cannot hit 60 FPS on this hardware/grid.");
        }
        // K-C verdict: the near-ring pan win on a portal scene under pure rotation.
        if warp_rot_fps >= 60.0 {
            println!(
                "ROTATION (portal ring pan): PASS — {warp_rot_fps:.1} FPS ≥60 FPS \
                      (portal columns pan from the near ring; zero forced recast)."
            );
        } else {
            println!("ROTATION (portal ring pan): FAIL — {warp_rot_fps:.1} FPS <60 FPS.");
        }
        println!("=================================================================");
    } // end !skip_warp
}
