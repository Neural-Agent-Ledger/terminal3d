//! `voxscene` — load the 5 procedurally-generated `.vox` building models (see
//! `re_voxscene::genmodels`), lay them out on a town grid with streets,
//! collision-check the assembly, convert it to a `re_world::Map`, and render
//! TWO ASCII views with our own raycaster.
//!
//! Prints: a per-model size table, the layout, the collision report, and both
//! ASCII frames; also SAVES the frames to `docs/town_preview.txt`.
//!
//! Run from the workspace root: `cargo run -p re_voxscene`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use re_voxscene::{
    assemble_town, layout_town, load_generated_town, overview_camera, render_ascii,
    street_level_camera, town_to_map, Town,
};

/// Framebuffer size for both ASCII previews (reasonably large, per the brief).
const COLS: usize = 120;
const ROWS: usize = 40;
/// Town block grid: 3 columns → 2 rows for 5 buildings.
const GRID_COLS: u32 = 3;
/// Street margin (voxel gap) between blocks AND around the outside.
const STREET: u32 = 6;

fn main() -> ExitCode {
    // Locate the workspace root (the dir holding `authoring/`). Works whether run
    // from the workspace root or the crate dir.
    let root = workspace_root();
    let authoring = root.join("authoring");

    println!("voxscene — generated-town voxel assembler + ASCII previewer");
    println!("authoring root: {}\n", authoring.display());

    // ── Load + report per-model sizes ──
    let buildings = match load_generated_town(&authoring) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR loading models: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("==== MODEL SIZES ({} models) ====", buildings.len());
    println!(
        "  {:<18} {:>6} {:>6} {:>6}   {:>12}",
        "model", "x", "y", "z", "solid voxels"
    );
    for b in &buildings {
        println!(
            "  {:<18} {:>6} {:>6} {:>6}   {:>12}",
            b.name,
            b.footprint_x(),
            b.footprint_y(),
            b.height(),
            b.solid_count()
        );
    }
    println!();

    // ── Deterministic layout ──
    let layout = layout_town(&buildings, GRID_COLS, STREET);
    println!(
        "==== LAYOUT ====\n  {} blocks ({}×{}), street margin {} voxels, town ground {}×{} voxels\n",
        layout.placements.len(),
        layout.blocks.0,
        layout.blocks.1,
        layout.street,
        layout.town_x,
        layout.town_y,
    );

    // ── Assemble + collision check (FAILS LOUD on overlap) ──
    let town = match assemble_town(buildings, layout) {
        Ok(t) => t,
        Err(e) => {
            // Print the full report if it was a collision; otherwise the message.
            if let re_voxscene::VoxError::Collision(report) = &e {
                println!("{}", report.render());
            }
            eprintln!("ERROR assembling town: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("{}", town.report.render());
    println!();

    // ── Convert to our world ──
    let map = town_to_map(&town);
    let (sx, sy, sz) = map.size();
    println!(
        "==== WORLD MAP ====\n  {sx}×{sy}×{sz} cells   floor slab at z=0, buildings z∈[1,{}]   sky open\n",
        sz - 1
    );

    // ── Render TWO views ──
    let street_cam = street_level_camera(&town);
    let over_cam = overview_camera(&town);

    let street_frame = render_ascii(&map, &street_cam, COLS, ROWS);
    let over_frame = render_ascii(&map, &over_cam, COLS, ROWS);

    let header_street = format!(
        "==== VIEW A: STREET-LEVEL (first-person, looking +y down a street) ====\n\
         camera pos=({:.1},{:.1},{:.1}) yaw={:.2} pitch={:.2}  fb={}×{}\n",
        street_cam.pos.x,
        street_cam.pos.y,
        street_cam.pos.z,
        street_cam.yaw,
        street_cam.pitch,
        COLS,
        ROWS
    );
    let header_over = format!(
        "==== VIEW B: ELEVATED OVERVIEW (pitched down over the rooftops) ====\n\
         camera pos=({:.1},{:.1},{:.1}) yaw={:.2} pitch={:.2}  fb={}×{}\n",
        over_cam.pos.x, over_cam.pos.y, over_cam.pos.z, over_cam.yaw, over_cam.pitch, COLS, ROWS
    );

    print!("{header_street}{street_frame}");
    println!();
    print!("{header_over}{over_frame}");
    println!();

    // ── Persist both frames ──
    let out = build_preview_file(
        &town,
        &header_street,
        &street_frame,
        &header_over,
        &over_frame,
    );
    let docs = root.join("docs");
    let _ = std::fs::create_dir_all(&docs);
    let preview_path = docs.join("town_preview.txt");
    match std::fs::write(&preview_path, out) {
        Ok(()) => println!(
            "saved both frames + collision report to {}",
            preview_path.display()
        ),
        Err(e) => eprintln!("WARNING: could not write {}: {e}", preview_path.display()),
    }

    ExitCode::SUCCESS
}

/// Assemble the persisted preview file (collision report + both frames).
fn build_preview_file(
    town: &Town,
    header_street: &str,
    street_frame: &str,
    header_over: &str,
    over_frame: &str,
) -> String {
    let mut s = String::new();
    s.push_str("terminal3d — generated-town voxel preview\n");
    s.push_str("(generated by `cargo run -p re_voxscene`)\n\n");
    s.push_str("==== MODEL SIZES ====\n");
    for b in &town.buildings {
        s.push_str(&format!(
            "  {:<18} {:>3}×{:>3}×{:>3}   {} voxels\n",
            b.name,
            b.footprint_x(),
            b.footprint_y(),
            b.height(),
            b.solid_count()
        ));
    }
    s.push('\n');
    s.push_str(&town.report.render());
    s.push('\n');
    s.push_str(header_street);
    s.push_str(street_frame);
    s.push('\n');
    s.push_str(header_over);
    s.push_str(over_frame);
    s
}

/// Find the workspace root by walking up from the current dir / exe until a dir
/// containing `authoring/voxelmodels` is found. Falls back to the cwd.
fn workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if dir.join("authoring").join("voxelmodels").is_dir() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    // Last resort: assume cwd is the workspace root.
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[allow(dead_code)]
fn _assert_path_type(_: &Path) {}
