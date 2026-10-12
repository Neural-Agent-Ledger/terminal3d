//! `gen_texfixture` — write the procedurally-generated image-sampler test fixtures.
//!
//! The `.lvl` image-texture sampler's only golden anchors used to bind a
//! copyrighted PNG that does not ship with the game; deleting it silently dropped
//! all sampler coverage instead of failing. `levels/TX5-genplate.lvl` binds a free,
//! generated plate instead — this is the generator, committed so the asset is
//! reproducible from source rather than an opaque blob.
//!
//! The animated `[texture] frames = ` path had the same problem with a real,
//! untracked, unknown-origin GIF (`vgif-ru-23830.gif`) — `--anim` generates its free
//! replacement the same way.
//!
//! Usage (from the workspace root):
//!
//! ```sh
//! cargo run -p re_voximport --bin gen_texfixture -- assets/texture/genplate_tex.png
//! cargo run -p re_voximport --bin gen_texfixture -- --anim assets/texture/gen_anim.gif
//! ```
//!
//! With no path argument, each mode writes to its own default path below, relative
//! to the CWD. Output is byte-stable: [`re_voximport::texfixture::plate_rgba`] and
//! [`re_voximport::texfixture::anim_frame_rgba`] are integer-only and take no input,
//! so re-running never produces a diff. The unit tests
//! `texfixture::tests::the_committed_png_matches_the_generator` and
//! `texfixture::tests::the_committed_gif_matches_the_generator` fail if a committed
//! asset and its generator ever disagree.
//!
//! Offline authoring tool: writes one file and prints a one-line report. No
//! terminal control, no renderer.

use re_voximport::texfixture;

/// Where the plate lives in the repository when no path is given.
const DEFAULT_PNG_OUT: &str = "assets/texture/genplate_tex.png";
/// Where the animated clip lives in the repository when no path is given.
const DEFAULT_GIF_OUT: &str = "assets/texture/gen_anim.gif";

fn create_parent(out: &str) -> Result<(), String> {
    let Some(parent) = std::path::Path::new(out).parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "gen_texfixture — write the generated image-sampler test fixtures\n\n\
             USAGE:\n  \
             gen_texfixture [OUT.png]          (default: {DEFAULT_PNG_OUT})\n  \
             gen_texfixture --anim [OUT.gif]   (default: {DEFAULT_GIF_OUT})"
        );
        return std::process::ExitCode::SUCCESS;
    }

    if args.first().is_some_and(|a| a == "--anim") {
        if args.len() > 2 {
            eprintln!(
                "gen_texfixture --anim: expected at most one output path, got {}",
                args.len() - 1
            );
            return std::process::ExitCode::FAILURE;
        }
        let out = args.get(1).map_or(DEFAULT_GIF_OUT, String::as_str);
        let gif = match texfixture::anim_gif() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gen_texfixture: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        if let Err(e) = create_parent(out) {
            eprintln!("gen_texfixture: {e}");
            return std::process::ExitCode::FAILURE;
        }
        if let Err(e) = std::fs::write(out, &gif) {
            eprintln!("gen_texfixture: cannot write {out}: {e}");
            return std::process::ExitCode::FAILURE;
        }
        println!(
            "gen_texfixture: wrote {out} ({}x{} RGBA8, {} frames, {} bytes)",
            texfixture::ANIM_W,
            texfixture::ANIM_H,
            texfixture::ANIM_FRAMES,
            gif.len()
        );
        return std::process::ExitCode::SUCCESS;
    }

    if args.len() > 1 {
        eprintln!(
            "gen_texfixture: expected at most one output path, got {}",
            args.len()
        );
        return std::process::ExitCode::FAILURE;
    }
    let out = args.first().map_or(DEFAULT_PNG_OUT, String::as_str);

    let png = match texfixture::plate_png() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gen_texfixture: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Err(e) = create_parent(out) {
        eprintln!("gen_texfixture: {e}");
        return std::process::ExitCode::FAILURE;
    }
    if let Err(e) = std::fs::write(out, &png) {
        eprintln!("gen_texfixture: cannot write {out}: {e}");
        return std::process::ExitCode::FAILURE;
    }
    println!(
        "gen_texfixture: wrote {out} ({}x{} RGBA8, {} bytes)",
        texfixture::PLATE_SIZE,
        texfixture::PLATE_SIZE,
        png.len()
    );
    std::process::ExitCode::SUCCESS
}
