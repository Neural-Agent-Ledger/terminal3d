//! `vox2lvl` — `.vox` + `.toml` → `.lvl` on stdout.
//!
//! Usage: `vox2lvl <level.vox> <level.toml> > level.lvl`
//!
//! Reads the binary voxel model and its companion metadata, derives the
//! canonical level via [`re_voximport::vox_to_level`], and prints the resulting
//! `.lvl` text to stdout. Any error goes to stderr with a non-zero exit code.

use std::process::ExitCode;

fn run() -> Result<String, String> {
    let mut args = std::env::args().skip(1);
    let vox_path = args
        .next()
        .ok_or_else(|| "usage: vox2lvl <level.vox> <level.toml>".to_string())?;
    let toml_path = args
        .next()
        .ok_or_else(|| "usage: vox2lvl <level.vox> <level.toml>".to_string())?;

    let vox_bytes = std::fs::read(&vox_path).map_err(|e| format!("cannot read {vox_path}: {e}"))?;
    let toml_str =
        std::fs::read_to_string(&toml_path).map_err(|e| format!("cannot read {toml_path}: {e}"))?;

    let level = re_voximport::vox_to_level(&vox_bytes, &toml_str).map_err(|e| e.to_string())?;
    Ok(re_voximport::to_lvl_text(&level))
}

fn main() -> ExitCode {
    match run() {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("vox2lvl: error: {e}");
            ExitCode::FAILURE
        }
    }
}
