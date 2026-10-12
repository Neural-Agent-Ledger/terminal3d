//! `renderqa` — the render-quality linter binary.
//!
//! Drives: load each level → static prologue (serial) → orbit poses → parallel
//! raycast + analyze → orbit shimmer → selective HTML photos → ranked
//! `report.html` + `report.json` + a console summary with a timing breakdown.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use re_core::Framebuffer;
use re_render::raycast;

use re_imgqa::html::{self, ReportRow};
use re_imgqa::metrics::orbit_shimmer;
use re_imgqa::orbit::{self, OrbitParams, OrbitPose, PitchMode};
use re_imgqa::run::{run_level, FrameResult};
use re_imgqa::static_map::analyze_static;
use re_imgqa::text_bind;
use re_imgqa::timing::{fmt_ms, ScopedTimer, StageTimers, Timing};

/// Photo-write policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PhotoPolicy {
    /// Degenerate + worst-N + spawn + an even-stride sample.
    Auto,
    /// Degenerate + worst-N + spawn only.
    Worst,
    /// One photo per analyzed frame.
    All,
}

struct Config {
    levels: Vec<PathBuf>,
    out: PathBuf,
    cols: usize,
    rows: usize,
    positions: usize,
    yaws: usize,
    pitch: PitchMode,
    photos: PhotoPolicy,
    worst_n: usize,
    threads: usize,
    open: bool,
}

fn main() -> ExitCode {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("renderqa: {e}\n");
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match run(cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("renderqa: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
usage: renderqa [LEVELS...]
  --out <dir>                 output dir (default target/renderqa)
  --grid <PRESET|COLSxROWS>   fhd=240x70 (default), qhd=320x90, 4k=480x135
  --positions <N>             position sample budget per level (default 40)
  --yaws <N>                  yaw directions per position (default 16)
  --no-pitch                  level pitch only (default 3: level/up/down)
  --photos <auto|worst|all>   photo-write policy (default auto)
  --worst-n <N>               worst frames photographed per level (default 20)
  --threads <N>               worker threads (default: available parallelism)
  --open                      open report.html when done";

fn parse_args() -> Result<Config, String> {
    let mut levels: Vec<PathBuf> = Vec::new();
    let mut out = PathBuf::from("target/renderqa");
    let (mut cols, mut rows) = (240usize, 70usize);
    let mut positions = 40usize;
    let mut yaws = 16usize;
    let mut pitch = PitchMode::Three;
    let mut photos = PhotoPolicy::Auto;
    let mut worst_n = 20usize;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut open = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(need(&mut args, "--out")?),
            "--grid" => {
                let (c, r) = parse_grid(&need(&mut args, "--grid")?)?;
                cols = c;
                rows = r;
            }
            "--positions" => {
                positions = need(&mut args, "--positions")?
                    .parse()
                    .map_err(|_| "bad --positions")?
            }
            "--yaws" => {
                yaws = need(&mut args, "--yaws")?
                    .parse()
                    .map_err(|_| "bad --yaws")?
            }
            "--no-pitch" => pitch = PitchMode::LevelOnly,
            "--photos" => {
                photos = match need(&mut args, "--photos")?.as_str() {
                    "auto" => PhotoPolicy::Auto,
                    "worst" => PhotoPolicy::Worst,
                    "all" => PhotoPolicy::All,
                    other => return Err(format!("bad --photos value: {other}")),
                }
            }
            "--worst-n" => {
                worst_n = need(&mut args, "--worst-n")?
                    .parse()
                    .map_err(|_| "bad --worst-n")?
            }
            "--threads" => {
                threads = need(&mut args, "--threads")?
                    .parse()
                    .map_err(|_| "bad --threads")?
            }
            "--open" => open = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with("--") => return Err(format!("unknown flag: {s}")),
            s => levels.push(PathBuf::from(s)),
        }
    }

    if levels.is_empty() {
        levels = default_levels();
        if levels.is_empty() {
            return Err("no levels given and no levels/*.lvl found".into());
        }
    }

    Ok(Config {
        levels,
        out,
        cols,
        rows,
        positions,
        yaws,
        pitch,
        photos,
        worst_n,
        threads: threads.max(1),
        open,
    })
}

fn need(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_grid(s: &str) -> Result<(usize, usize), String> {
    match s {
        "fhd" => Ok((240, 70)),
        "qhd" => Ok((320, 90)),
        "4k" => Ok((480, 135)),
        _ => {
            let (c, r) = s
                .split_once(['x', 'X'])
                .ok_or_else(|| format!("bad --grid: {s} (want PRESET or COLSxROWS)"))?;
            let c = c.parse().map_err(|_| "bad grid cols")?;
            let r = r.parse().map_err(|_| "bad grid rows")?;
            Ok((c, r))
        }
    }
}

fn default_levels() -> Vec<PathBuf> {
    let dir = Path::new("levels");
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "lvl").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn run(cfg: Config) -> Result<(), String> {
    let wall_start = Instant::now();
    std::fs::create_dir_all(&cfg.out)
        .map_err(|e| format!("cannot create {}: {e}", cfg.out.display()))?;

    let timers = StageTimers::new();
    let params = OrbitParams {
        positions: cfg.positions,
        yaws: cfg.yaws,
        pitch: cfg.pitch,
    };

    let mut all_rows: Vec<ReportRow> = Vec::new();
    let mut statics = Vec::new();
    let mut shimmer_summary: Vec<(String, f32)> = Vec::new();
    let mut per_level_ns: Vec<(String, u64)> = Vec::new();
    let mut total_frames = 0u64;
    let mut total_photos = 0u64;
    let mut worst_global: Vec<(String, String, f32)> = Vec::new(); // (level, pose, sep)

    for level_path in &cfg.levels {
        let lvl_start = Instant::now();
        let level = re_level::load(level_path)
            .map_err(|e| format!("failed to load {}: {e}", level_path.display()))?;
        let level_name = if level.name.is_empty() {
            level_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("level")
                .to_string()
        } else {
            level.name.clone()
        };
        let mut map = level.build_map();

        // ── bind text surfaces (headless replica of the host game's level
        // text binder — see `re_imgqa::text_bind`) ──
        //
        // Publishes ONE registry per level, for exactly this level's render
        // section below. Safe under `run_level`'s `std::thread::scope` fan-out
        // because this outer `for` loop processes levels strictly serially —
        // only the POSES of the CURRENT level render in parallel, and those
        // worker threads only READ the already-installed registry; nothing
        // writes it again until the next loop iteration. See `text_bind`'s
        // module docs for the full argument, and `re_render::scene_text`'s
        // for why the registry is a process-global `RwLock` in the first
        // place (installed by level-load, read by render, possibly different
        // threads).
        let (text_registry, note_report) = text_bind::bind_level_text(&level, &mut map);
        if !note_report.unplaced.is_empty() {
            eprintln!(
                "renderqa: {level_name} — {} note(s) did not resolve to a wall face: {:?}",
                note_report.unplaced.len(),
                note_report.unplaced
            );
        }
        let map = map; // freeze: shared immutably by the parallel section below
        let _text_scope = re_render::SceneTextScope::new(text_registry);

        // ── static prologue (serial, no raycast) ──
        let st = analyze_static(&level, &map);

        // ── orbit poses ──
        let spawn = level.spawn_camera();
        let poses: Vec<OrbitPose> = orbit::generate(&map, spawn, &params);

        // ── parallel raycast + analyze ──
        let results: Vec<FrameResult> =
            run_level(&map, &poses, cfg.cols, cfg.rows, cfg.threads, &timers);
        total_frames += results.len() as u64;

        // ── orbit shimmer (serial; needs angular neighbours) ──
        let level_shimmer = compute_shimmer(&map, &poses, &params, cfg.cols, cfg.rows);
        shimmer_summary.push((level_name.clone(), level_shimmer));

        // ── photo selection ──
        let level_dir = cfg.out.join(sanitize(&level_name));
        let to_photo = select_photos(&results, cfg.photos, cfg.worst_n);
        if !to_photo.is_empty() {
            std::fs::create_dir_all(&level_dir)
                .map_err(|e| format!("cannot create {}: {e}", level_dir.display()))?;
        }

        // Render selected photos (re-raycast each chosen pose; cheap relative to
        // the full fan-out). Timed under the html stage.
        let pose_by_index: Vec<&OrbitPose> = poses.iter().collect();
        let mut photo_for: std::collections::HashMap<usize, String> =
            std::collections::HashMap::new();
        let mut fb = Framebuffer::new(cfg.cols, cfg.rows);
        for &res_i in &to_photo {
            let res = &results[res_i];
            let op = pose_by_index[res.index];
            fb.clear();
            raycast(&map, &op.pose, &mut fb);
            let t = ScopedTimer::start();
            let html_doc = html::frame_photo(&level_name, &res.id, &fb, &res.metrics);
            let fname = format!("{}.html", sanitize(&res.id));
            let path = level_dir.join(&fname);
            std::fs::write(&path, html_doc)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            timers.add_html(t.stop());
            photo_for.insert(res.index, format!("{}/{}", sanitize(&level_name), fname));
            total_photos += 1;
        }

        // ── rows for the report ──
        for res in &results {
            all_rows.push(ReportRow {
                level: level_name.clone(),
                pose_id: res.id.clone(),
                metrics: res.metrics.clone(),
                photo_rel: photo_for.get(&res.index).cloned(),
            });
            worst_global.push((level_name.clone(), res.id.clone(), res.metrics.separation));
        }

        statics.push(st);
        per_level_ns.push((level_name.clone(), lvl_start.elapsed().as_nanos() as u64));
        let _ = lvl_start;
    }

    // ── assemble timing ──
    let timing = Timing {
        wall_ns: wall_start.elapsed().as_nanos() as u64,
        per_level_ns,
        stages: timers.snapshot(),
        frames_analyzed: total_frames,
        photos_written: total_photos,
    };

    // ── write report.html + report.json ──
    let report_html_doc = html::report_html(&all_rows, &statics, &shimmer_summary, &timing);
    let report_json_doc = html::report_json(&all_rows, &statics, &shimmer_summary, &timing);
    let html_path = cfg.out.join("report.html");
    let json_path = cfg.out.join("report.json");
    std::fs::write(&html_path, report_html_doc)
        .map_err(|e| format!("cannot write report.html: {e}"))?;
    std::fs::write(&json_path, report_json_doc)
        .map_err(|e| format!("cannot write report.json: {e}"))?;

    // ── console summary ──
    print_summary(&cfg, &timing, &worst_global, &html_path);

    if cfg.open {
        open_in_browser(&html_path);
    }
    Ok(())
}

/// Pick which frame results get an HTML photo, per the policy. Returns indices
/// into `results`.
fn select_photos(results: &[FrameResult], policy: PhotoPolicy, worst_n: usize) -> Vec<usize> {
    let mut chosen: HashSet<usize> = HashSet::new();

    if policy == PhotoPolicy::All {
        return (0..results.len()).collect();
    }

    // (a) every degenerate / flagged frame.
    for (i, r) in results.iter().enumerate() {
        let m = &r.metrics;
        if m.is_blank || m.single_glyph || m.single_color || m.invisible_glyph_pct > 25.0 {
            chosen.insert(i);
        }
    }
    // (b) worst-N by separation (ascending = worst first).
    let mut by_sep: Vec<usize> = (0..results.len()).collect();
    by_sep.sort_by(|&a, &b| {
        results[a]
            .metrics
            .separation
            .partial_cmp(&results[b].metrics.separation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in by_sep.iter().take(worst_n) {
        chosen.insert(i);
    }
    // (c) every spawn pose.
    for (i, r) in results.iter().enumerate() {
        if r.is_spawn {
            chosen.insert(i);
        }
    }
    // (d) auto: an even-stride representative sample (~1 per position-ish).
    if policy == PhotoPolicy::Auto {
        let stride = (results.len() / 64).max(1);
        for i in (0..results.len()).step_by(stride) {
            chosen.insert(i);
        }
    }

    let mut out: Vec<usize> = chosen.into_iter().collect();
    out.sort_unstable();
    out
}

/// Compute the mean shimmer over a level's orbit by re-raycasting each shimmer
/// group's yaw sequence and pairing angular neighbours.
fn compute_shimmer(
    map: &re_world::Map,
    poses: &[OrbitPose],
    params: &OrbitParams,
    cols: usize,
    rows: usize,
) -> f32 {
    let groups = orbit::shimmer_groups(poses, params);
    if groups.is_empty() {
        return 0.0;
    }
    let pose_by_index: Vec<&OrbitPose> = poses.iter().collect();
    let mut sum = 0.0f64;
    let mut groups_counted = 0u32;
    for (yaw_step, idxs) in groups {
        // Raycast each yaw frame for this group.
        let mut frames: Vec<Framebuffer> = Vec::with_capacity(idxs.len());
        for &gi in &idxs {
            let mut fb = Framebuffer::new(cols, rows);
            raycast(map, &pose_by_index[gi].pose, &mut fb);
            frames.push(fb);
        }
        let refs: Vec<&Framebuffer> = frames.iter().collect();
        let o = orbit_shimmer(&refs, yaw_step);
        if o.pairs > 0 {
            sum += o.shimmer as f64;
            groups_counted += 1;
        }
    }
    if groups_counted == 0 {
        0.0
    } else {
        (sum / groups_counted as f64) as f32
    }
}

fn print_summary(cfg: &Config, timing: &Timing, worst: &[(String, String, f32)], html_path: &Path) {
    let mut worst = worst.to_vec();
    worst.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("── renderqa summary ──────────────────────────────");
    println!("levels scanned : {}", cfg.levels.len());
    println!("grid           : {}x{}", cfg.cols, cfg.rows);
    println!("threads        : {}", cfg.threads);
    println!("frames analyzed: {}", timing.frames_analyzed);
    println!("photos written : {}", timing.photos_written);
    println!("frames/sec     : {:.0}", timing.frames_per_sec());
    println!("wall time      : {}", fmt_ms(timing.wall_ns));
    println!(
        "stage breakdown: raycast {} · analyze {} · html {}",
        fmt_ms(timing.stages.raycast_ns),
        fmt_ms(timing.stages.analyze_ns),
        fmt_ms(timing.stages.html_ns),
    );
    println!("worst 3 frames (lowest separation):");
    for (lvl, pose, sep) in worst.iter().take(3) {
        println!("  {sep:>7.3}  {lvl} · {pose}");
    }
    println!("report         : {}", html_path.display());
    println!("──────────────────────────────────────────────────");
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn open_in_browser(path: &Path) {
    let p = path.to_string_lossy().to_string();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", &p])
        .spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(&p).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(&p).spawn();
}
