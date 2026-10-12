//! `lvl2vox` — `.lvl` → `.vox` + `.toml` reverse converter.
//!
//! Usage: `lvl2vox <level.lvl> <out.vox> [out.toml]`
//!
//! Loads a terminal3d `.lvl` file, stamps its geometry and puzzle graph into a
//! colour-tagged voxel grid (using the SAME tag→RGB convention as `vox2lvl`),
//! and writes:
//!
//! - `<out.vox>` — a MagicaVoxel-compatible `.vox` file readable by Vengi,
//!   MagicaVoxel, or Goxel. The palette uses the canonical 13-tag RGBs so
//!   `vox2lvl` re-derives the level correctly.
//! - `<out.toml>` (optional, defaults to `<out>.toml` next to the `.vox`) —
//!   the companion metadata sidecar containing all non-spatial fields.
//!
//! After editing the `.vox` in a voxel editor, run:
//!   `vox2lvl <out.vox> <out.toml> > edited.lvl`
//! to convert back to the canonical `.lvl` format.
//!
//! ## Round-trip fidelity (see `re_voximport::level_to_toml` for the full table)
//!
//! Lossless: room size, node positions + roles, pipe paths + switch cells,
//! door/player/pedestal positions, decal regions, node names/glyphs/pools,
//! inputs, effects, palette fog, materials.
//!
//! Lossy (honest): `[part]`/`[emit]`/`[note]`/`[holo]`/`[vox]` sections are
//! NOT preserved (they are in the original `.lvl`, not in the `.toml` sidecar);
//! pipe density/color/state revert to defaults on re-import; pedestal box
//! dimensions collapse to 1×1×1 at the center cell.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let lvl_path = args
        .next()
        .ok_or_else(|| "usage: lvl2vox <level.lvl> <out.vox> [out.toml]".to_string())?;
    let vox_path = args
        .next()
        .ok_or_else(|| "usage: lvl2vox <level.lvl> <out.vox> [out.toml]".to_string())?;
    // Default toml path: same stem as the .vox, but with .toml extension.
    let toml_path: PathBuf = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            let base = Path::new(&vox_path).with_extension("toml");
            base
        }
    };

    // Load the .lvl.
    let level = re_level::load(&lvl_path).map_err(|e| format!("cannot load '{lvl_path}': {e}"))?;

    // Encode to .vox bytes.
    let vox_bytes = re_voximport::level_to_vox_bytes(&level);

    // Verify the output is a valid .vox (dot_vox round-trip sanity check).
    dot_vox::load_bytes(&vox_bytes)
        .map_err(|e| format!("internal: emitted .vox failed validation: {e}"))?;

    // Write the .vox file.
    std::fs::write(&vox_path, &vox_bytes).map_err(|e| format!("cannot write '{vox_path}': {e}"))?;

    // Emit and write the companion .toml.
    let toml_str = re_voximport::level_to_toml(&level);
    std::fs::write(&toml_path, toml_str.as_bytes())
        .map_err(|e| format!("cannot write '{}': {e}", toml_path.display()))?;

    // Report what was written.
    let node_count = level.nodes.len();
    let pipe_count = level.pipes.len();
    let vox_cell_count = {
        // Re-parse to count voxels emitted (cheap — already in memory).
        dot_vox::load_bytes(&vox_bytes)
            .map(|d| d.models.first().map(|m| m.voxels.len()).unwrap_or(0))
            .unwrap_or(0)
    };
    eprintln!(
        "lvl2vox: wrote {vox_path} ({vox_cell_count} voxels, {node_count} nodes, {pipe_count} pipes)"
    );
    eprintln!("lvl2vox: wrote {}", toml_path.display());
    eprintln!(
        "lvl2vox: to convert back: vox2lvl {vox_path} {} > out.lvl",
        toml_path.display()
    );

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lvl2vox: error: {e}");
            ExitCode::FAILURE
        }
    }
}
