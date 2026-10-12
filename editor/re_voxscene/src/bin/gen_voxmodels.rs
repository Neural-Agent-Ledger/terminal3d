//! `gen_voxmodels` — write the procedurally-generated `.vox` building models
//! that replace a removed, unlicensed third-party voxel kit.
//!
//! Usage (from the workspace root):
//!
//! ```sh
//! cargo run -p re_voxscene --bin gen_voxmodels
//! ```
//!
//! With no arguments, writes all 5 models to their default committed paths
//! under `authoring/voxelmodels/generated/`. Pass an explicit output directory
//! to write elsewhere:
//!
//! ```sh
//! cargo run -p re_voxscene --bin gen_voxmodels -- /some/other/dir
//! ```
//!
//! Output is byte-stable: [`re_voxscene::genmodels::generated_models`] takes no
//! input and is pure integer arithmetic, so re-running never produces a diff.
//! `genmodels::tests::every_committed_vox_matches_its_generator` fails if a
//! committed asset and its generator ever disagree.
//!
//! Offline authoring tool: writes files and prints a one-line report per model.
//! No terminal control, no renderer.

use re_voxscene::genmodels;

/// Where the generated models live in the repository when no directory is given.
const DEFAULT_OUT_DIR: &str = "authoring/voxelmodels/generated";

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "gen_voxmodels — write the generated replacement .vox building models\n\n\
             USAGE:\n  \
             gen_voxmodels [OUT_DIR]   (default: {DEFAULT_OUT_DIR})"
        );
        return std::process::ExitCode::SUCCESS;
    }
    if args.len() > 1 {
        eprintln!(
            "gen_voxmodels: expected at most one output directory, got {}",
            args.len()
        );
        return std::process::ExitCode::FAILURE;
    }
    let out_dir = args.first().map_or(DEFAULT_OUT_DIR, String::as_str);
    let out_dir = std::path::Path::new(out_dir);

    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("gen_voxmodels: cannot create {}: {e}", out_dir.display());
        return std::process::ExitCode::FAILURE;
    }

    let models = genmodels::generated_models();
    println!(
        "gen_voxmodels: writing {} model(s) to {}",
        models.len(),
        out_dir.display()
    );
    for m in &models {
        let bytes = match m.write_bytes() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("gen_voxmodels: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let path = out_dir.join(format!("{}.vox", m.stem));
        if let Err(e) = std::fs::write(&path, &bytes) {
            eprintln!("gen_voxmodels: cannot write {}: {e}", path.display());
            return std::process::ExitCode::FAILURE;
        }
        let (w, d, h) = m.size;
        println!(
            "  {:<14} {}x{}x{}  {} solid voxels  {} bytes  -> {}",
            m.stem,
            w,
            d,
            h,
            m.voxels.len(),
            bytes.len(),
            path.display()
        );
    }
    std::process::ExitCode::SUCCESS
}
