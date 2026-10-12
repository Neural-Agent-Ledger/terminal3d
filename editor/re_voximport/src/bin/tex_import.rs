//! `tex_import` — turn a FOREIGN image (a downloaded texture, a photo, a normal
//! map) into something a `.lvl` can bind directly: resized, converted, and printed
//! as a ready-to-paste `[texture NAME]` block.
//!
//! Import CLI, following the same pattern as `gen_texfixture`: resize,
//! normal→height, GIF→frames, emit the `[texture]` block. The grammar this
//! tool emits is owned by `re_level` (`docs/MATERIALS.md` § "Image textures —
//! `[texture NAME]`" and § "Relief"): only `file`, `max_size`, `height` are
//! keys `re_level` parses today, so that is all this tool ever writes.
//!
//! ```text
//! tex_import <INPUT> [--out DIR] [--name NAME] [--max-size N]
//!            [--normal-to-height] [--frames] [--force]
//! ```
//!
//! # Why the resize happens at IMPORT time, not load time
//!
//! `re_level` already box-downsamples a `[texture] file` on load when `max_size`
//! is set (`decode_texture_file` / `cap_longest_edge` in `re_level`) — that
//! guards RAM, but the repo still carries the huge source file.
//! `tex_import` bakes the SAME cap into the file that ships, so a downloaded
//! 9428x3548 photo becomes a small PNG on disk, not just a small buffer in
//! memory. It re-implements the identical box/`Triangle` downsample math (same
//! rounding, same filter) so a level that later also sets `max_size` in the
//! `.lvl` is a defensive no-op, not a second, different reduction.
//!
//! # Why WebP is never WRITTEN
//!
//! `image` 0.25's `webp` feature (shared with `re_level`, see `Cargo.toml`) is a
//! decoder. `tex_import` reads WebP fine; every non-animated output this tool
//! writes is PNG (already the format `re_level`'s own doc examples show), so no
//! WebP encoder is ever needed. GIF is the one format that survives ONLY as
//! itself, because a GIF's frames are the entire point of importing one.
//!
//! # Why the normal→height math lives here and not in `re_level`
//!
//! A single row-wise (or column-wise) integration of `dh/dx = -nx/nz` drifts and
//! streaks — errors accumulate along one axis with nothing to correct them. This
//! tool integrates BOTH axes independently (row-wise from `dh/dx`, column-wise
//! from `dh/dy`) and averages the two fields, which is exactly why this
//! conversion is pushed out of the (per-frame) loader and into offline tooling:
//! it is an O(w·h) pass with two full sweeps, worth paying once at import time,
//! not on every level load.
//!
//! Offline authoring tool: writes files only under `--out`, prints a report and a
//! `.lvl` fragment. No terminal control, no renderer, no `unsafe`.

use image::codecs::gif::{GifDecoder, GifEncoder, Repeat};
use image::{AnimationDecoder, DynamicImage, Frame as GifFrame, GrayImage, RgbaImage};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return ExitCode::SUCCESS;
    }
    let args = match parse_args(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("tex_import: {e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tex_import: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "USAGE: tex_import <INPUT> [--out DIR] [--name NAME] [--max-size N] \
         [--normal-to-height] [--frames] [--force]"
    );
}

fn print_help() {
    println!("tex_import — foreign image -> .lvl-ready texture asset\n");
    print_usage();
    println!();
    println!("  --out DIR             output directory (default: INPUT's own directory)");
    println!("  --name NAME           output base name (default: INPUT's file stem)");
    println!(
        "  --max-size N          cap the longest edge at N px (box/triangle downsample, aspect kept)"
    );
    println!(
        "  --normal-to-height    treat INPUT as a tangent-space normal map; emit an 8-bit height map"
    );
    println!(
        "  --frames              report GIF frame count/fps and (if INPUT is a GIF) keep every frame"
    );
    println!("  --force               overwrite existing output files");
    println!();
    println!("Prints a ready-to-paste [texture NAME] block for a .lvl on stdout.");
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    input: PathBuf,
    out: Option<PathBuf>,
    name: Option<String>,
    max_size: Option<u32>,
    normal_to_height: bool,
    frames: bool,
    force: bool,
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut out = None;
    let mut name = None;
    let mut max_size = None;
    let mut normal_to_height = false;
    let mut frames = false;
    let mut force = false;

    let mut i = 0;
    while i < raw.len() {
        let a = raw[i].as_str();
        match a {
            "--out" => {
                i += 1;
                let v = raw.get(i).ok_or("--out requires a DIR value")?;
                out = Some(PathBuf::from(v));
            }
            "--name" => {
                i += 1;
                let v = raw.get(i).ok_or("--name requires a NAME value")?;
                name = Some(v.clone());
            }
            "--max-size" => {
                i += 1;
                let v = raw.get(i).ok_or("--max-size requires an N value")?;
                let n: u32 = v
                    .parse()
                    .map_err(|_| format!("--max-size: not a whole number: '{v}'"))?;
                if n == 0 {
                    return Err("--max-size must be >= 1 (0 would erase the image)".to_string());
                }
                max_size = Some(n);
            }
            "--normal-to-height" => normal_to_height = true,
            "--frames" => frames = true,
            "--force" => force = true,
            _ if a.starts_with('-') => return Err(format!("unknown flag: {a}")),
            _ => {
                if input.is_some() {
                    return Err(format!("unexpected extra positional argument: '{a}'"));
                }
                input = Some(PathBuf::from(a));
            }
        }
        i += 1;
    }

    let input = input.ok_or("missing INPUT image path")?;
    Ok(Args {
        input,
        out,
        name,
        max_size,
        normal_to_height,
        frames,
        force,
    })
}

// ---------------------------------------------------------------------------
// Top-level flow
// ---------------------------------------------------------------------------

fn run(args: &Args) -> Result<(), String> {
    if !args.input.is_file() {
        return Err(format!("input image not found: {}", args.input.display()));
    }

    let out_dir = args.out.clone().unwrap_or_else(|| {
        args.input
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    });
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create output dir {}: {e}", out_dir.display()))?;

    let name = match &args.name {
        Some(n) => n.clone(),
        None => args
            .input
            .file_stem()
            .and_then(OsStr::to_str)
            .map(str::to_string)
            .ok_or_else(|| format!("cannot derive a --name from {}", args.input.display()))?,
    };

    if args.normal_to_height {
        run_normal_to_height(args, &out_dir, &name)
    } else {
        run_resize_convert(args, &out_dir, &name)
    }
}

/// The default path: resize/convert the colour image, optionally preserving a
/// GIF's animation when `--frames` is given.
fn run_resize_convert(args: &Args, out_dir: &Path, name: &str) -> Result<(), String> {
    let gif_frames = open_gif_frames(&args.input).ok();

    if let Some(frames) = &gif_frames {
        report_gif_frames(frames);
    }

    if args.frames {
        if let Some(frames) = gif_frames {
            return write_animated_gif(args, out_dir, name, frames);
        }
        println!(
            "tex_import: --frames was given but {} does not decode as an animated GIF; \
             converting it as a single still image instead.",
            args.input.display()
        );
    } else if let Some(frames) = &gif_frames {
        println!(
            "tex_import: {} has {} frames; only the first is converted (pass --frames to \
             keep the animation).",
            args.input.display(),
            frames.len()
        );
    }

    write_static_image(args, out_dir, name)
}

fn write_static_image(args: &Args, out_dir: &Path, name: &str) -> Result<(), String> {
    let bytes =
        fs::read(&args.input).map_err(|e| format!("cannot read {}: {e}", args.input.display()))?;
    let dynimg = image::load_from_memory(&bytes)
        .map_err(|e| format!("cannot decode {}: {e}", args.input.display()))?;
    let (ow, oh) = (dynimg.width(), dynimg.height());
    let resized = cap_longest_edge(dynimg, args.max_size);
    let (nw, nh) = (resized.width(), resized.height());

    let out_path = out_dir.join(format!("{name}.png"));
    check_overwrite(std::slice::from_ref(&out_path), &args.input, args.force)?;
    resized
        .save(&out_path)
        .map_err(|e| format!("cannot write {}: {e}", out_path.display()))?;

    report_resize(ow, oh, nw, nh);
    println!("tex_import: wrote {}", out_path.display());
    emit_block(name, &out_path, args.max_size, None);
    Ok(())
}

fn write_animated_gif(
    args: &Args,
    out_dir: &Path,
    name: &str,
    frames: Vec<image::Frame>,
) -> Result<(), String> {
    let n_frames = frames.len();
    let (ow, oh) = {
        let first = frames.first().ok_or("GIF decoded with zero frames")?;
        (first.buffer().width(), first.buffer().height())
    };

    let out_path = out_dir.join(format!("{name}.gif"));
    check_overwrite(std::slice::from_ref(&out_path), &args.input, args.force)?;

    let mut resized_frames: Vec<image::Frame> = Vec::with_capacity(n_frames);
    let mut nw = ow;
    let mut nh = oh;
    for f in frames {
        let delay = f.delay();
        let buf = f.into_buffer();
        let resized = cap_longest_edge(DynamicImage::ImageRgba8(buf), args.max_size).to_rgba8();
        nw = resized.width();
        nh = resized.height();
        resized_frames.push(GifFrame::from_parts(resized, 0, 0, delay));
    }

    let file =
        File::create(&out_path).map_err(|e| format!("cannot write {}: {e}", out_path.display()))?;
    let mut encoder = GifEncoder::new(file);
    encoder
        .set_repeat(Repeat::Infinite)
        .map_err(|e| format!("cannot set GIF loop mode for {}: {e}", out_path.display()))?;
    encoder
        .encode_frames(resized_frames)
        .map_err(|e| format!("cannot encode {}: {e}", out_path.display()))?;

    report_resize(ow, oh, nw, nh);
    println!(
        "tex_import: wrote {} ({n_frames} frames, all preserved)",
        out_path.display()
    );
    emit_block(name, &out_path, args.max_size, None);
    Ok(())
}

/// `--normal-to-height`: derive an 8-bit relief map from a tangent-space normal
/// map by integrating its gradient field along both axes and averaging the two
/// passes. Also writes a resized colour copy of the SAME input so the emitted
/// block always parses on its own (see the printed NOTE — swap `file` for a real
/// diffuse texture when you have one; this tool processes one input at a time).
fn run_normal_to_height(args: &Args, out_dir: &Path, name: &str) -> Result<(), String> {
    let bytes =
        fs::read(&args.input).map_err(|e| format!("cannot read {}: {e}", args.input.display()))?;
    let dynimg = image::load_from_memory(&bytes)
        .map_err(|e| format!("cannot decode {}: {e}", args.input.display()))?;
    validate_normal_map(&dynimg)?;

    let (ow, oh) = (dynimg.width(), dynimg.height());
    let resized = cap_longest_edge(dynimg, args.max_size);
    let (nw, nh) = (resized.width(), resized.height());
    let rgba = resized.to_rgba8();

    let color_path = out_dir.join(format!("{name}.png"));
    let height_path = out_dir.join(format!("{name}_height.png"));
    check_overwrite(
        &[color_path.clone(), height_path.clone()],
        &args.input,
        args.force,
    )?;

    rgba.save(&color_path)
        .map_err(|e| format!("cannot write {}: {e}", color_path.display()))?;

    let heights = normal_to_height(&rgba);
    let gray = GrayImage::from_raw(nw, nh, heights)
        .ok_or("internal error: height buffer size did not match the image dimensions")?;
    gray.save(&height_path)
        .map_err(|e| format!("cannot write {}: {e}", height_path.display()))?;

    report_resize(ow, oh, nw, nh);
    println!(
        "tex_import: wrote {} (colour copy of the imported normal map)",
        color_path.display()
    );
    println!(
        "tex_import: wrote {} (8-bit height, integrated from the normal map's gradient field, \
         both axes averaged)",
        height_path.display()
    );
    println!(
        "tex_import: NOTE — 'file' below points at the imported normal map's OWN pixels; swap \
         it for your diffuse texture if you have one separately — this tool processes one input \
         at a time and only fills 'file' so the block below parses on its own."
    );
    emit_block(name, &color_path, args.max_size, Some(&height_path));
    Ok(())
}

// ---------------------------------------------------------------------------
// Resize (mirrors `re_level::cap_longest_edge` — same filter, same rounding, so
// baking the cap in at import time and re-checking it at load time agree)
// ---------------------------------------------------------------------------

fn cap_longest_edge(img: DynamicImage, max_size: Option<u32>) -> DynamicImage {
    match max_size {
        Some(cap) if cap > 0 && img.width().max(img.height()) > cap => {
            let (w, h) = (f64::from(img.width()), f64::from(img.height()));
            let scale = f64::from(cap) / w.max(h);
            let nw = ((w * scale).round() as u32).max(1);
            let nh = ((h * scale).round() as u32).max(1);
            img.resize_exact(nw, nh, image::imageops::FilterType::Triangle)
        }
        _ => img,
    }
}

fn rgba_byte_cost(w: u32, h: u32) -> u64 {
    u64::from(w) * u64::from(h) * 4
}

fn human_bytes(n: u64) -> String {
    let mb = n as f64 / (1024.0 * 1024.0);
    if mb >= 1.0 {
        format!("{mb:.2} MB")
    } else {
        format!("{:.1} KB", n as f64 / 1024.0)
    }
}

fn report_resize(ow: u32, oh: u32, nw: u32, nh: u32) {
    let ob = rgba_byte_cost(ow, oh);
    let nb = rgba_byte_cost(nw, nh);
    println!(
        "tex_import: {ow}x{oh} ({}, {ob} B RGBA) -> {nw}x{nh} ({}, {nb} B RGBA)",
        human_bytes(ob),
        human_bytes(nb)
    );
}

// ---------------------------------------------------------------------------
// GIF frame handling
// ---------------------------------------------------------------------------

fn open_gif_frames(path: &Path) -> Result<Vec<image::Frame>, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    let decoder = GifDecoder::new(BufReader::new(f)).map_err(|e| e.to_string())?;
    decoder
        .into_frames()
        .collect_frames()
        .map_err(|e| e.to_string())
}

/// Average delay -> fps, from the source GIF's own per-frame delays (not a
/// guessed constant).
fn average_fps(frames: &[image::Frame]) -> (f64, f64) {
    let mut total_ms = 0f64;
    let mut n = 0u32;
    for f in frames {
        let (num, den) = f.delay().numer_denom_ms();
        if den > 0 {
            total_ms += f64::from(num) / f64::from(den);
            n += 1;
        }
    }
    let avg_ms = if n > 0 { total_ms / f64::from(n) } else { 0.0 };
    let fps = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 };
    (avg_ms, fps)
}

fn report_gif_frames(frames: &[image::Frame]) {
    let (avg_ms, fps) = average_fps(frames);
    println!(
        "tex_import: source is animated: {} frames, ~{fps:.2} fps (avg delay {avg_ms:.1} ms)",
        frames.len()
    );
}

// ---------------------------------------------------------------------------
// Normal map -> height (gradient-field integration)
// ---------------------------------------------------------------------------

/// Clamp on the Z (blue) component before dividing, so a near-tangent or
/// slightly-out-of-range texel cannot blow the gradient up to +/-infinity.
const MIN_NZ: f32 = 0.06;

/// Integrate `dh/dx = -nx/nz` row-wise and `dh/dy = -ny/nz` column-wise
/// independently, then average the two fields, then normalise to `[0, 255]`.
///
/// A single-axis integration drifts (errors accumulate along the one direction
/// integrated, with nothing to correct them) and streaks along that axis; taking
/// both and averaging cancels most of that per-axis bias.
///
/// `dhdx`/`dhdy` name the two partial-derivative fields this integrates
/// (`dh/dx`, `dh/dy`) — clippy's `similar_names` lint flags them as
/// look-alikes, but they ARE the two distinct axes by design.
#[allow(clippy::similar_names)]
fn normal_to_height(img: &RgbaImage) -> Vec<u8> {
    // Below this range floor the integrated field is treated as flat (see the
    // comment at its use below); declared up front, not mixed in after other
    // statements.
    const FLAT_RANGE_FLOOR: f32 = 1.0;

    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut dhdx = vec![0f32; w * h];
    let mut dhdy = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            let nx = f32::from(p[0]) / 255.0 * 2.0 - 1.0;
            let ny = f32::from(p[1]) / 255.0 * 2.0 - 1.0;
            let nz = (f32::from(p[2]) / 255.0 * 2.0 - 1.0).max(MIN_NZ);
            dhdx[y * w + x] = -nx / nz;
            dhdy[y * w + x] = -ny / nz;
        }
    }

    let mut h_row = vec![0f32; w * h];
    for y in 0..h {
        let mut acc = 0f32;
        for x in 0..w {
            if x > 0 {
                acc += dhdx[y * w + x];
            }
            h_row[y * w + x] = acc;
        }
    }

    let mut h_col = vec![0f32; w * h];
    for x in 0..w {
        let mut acc = 0f32;
        for y in 0..h {
            if y > 0 {
                acc += dhdy[y * w + x];
            }
            h_col[y * w + x] = acc;
        }
    }

    let mut combined = vec![0f32; w * h];
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..w * h {
        let v = f32::midpoint(h_row[i], h_col[i]);
        combined[i] = v;
        lo = lo.min(v);
        hi = hi.max(v);
    }

    // A GENUINELY flat map (every texel encodes the same near-(0,0,1) normal)
    // still carries a tiny nonzero gradient under 8-bit quantization — the
    // conventional "flat" encoding (128, 128, 255) decodes to `nx = ny =
    // 1/255`, not exactly 0, so the integrated field drifts by a few
    // thousandths of a unit per pixel of width/height. Min-max normalization
    // would stretch even that noise floor to the FULL 0..255 range, painting a
    // visible fake ramp onto a surface that has no real relief. Below this
    // floor the field is treated as flat and emitted as a constant mid-grey
    // instead of being amplified.
    if hi - lo < FLAT_RANGE_FLOOR {
        return vec![128u8; w * h];
    }

    let range = hi - lo;
    combined
        .iter()
        .map(|&v| (((v - lo) / range) * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// A cheap, honest heuristic — not a guarantee — that `img` looks like a
/// tangent-space normal map: it must have real colour channels (not grayscale),
/// and the Z (blue) channel must dominate, since a valid tangent-space normal
/// points mostly outward (`nz >= 0`, encoded as blue >= 128) almost everywhere.
/// A photo has no such bias, so this catches the "ran --normal-to-height on a
/// photo by mistake" case without pretending to be a real classifier.
fn validate_normal_map(img: &DynamicImage) -> Result<(), String> {
    const MAX_LOW_BLUE_FRACTION: f64 = 0.30;

    if !img.color().has_color() {
        return Err(
            "does not look like a tangent-space normal map: the source decoded as grayscale \
             (no distinct R/G/B channels to read as X/Y/Z)"
                .to_string(),
        );
    }
    let rgba = img.to_rgba8();
    let mut total: u64 = 0;
    let mut low_blue: u64 = 0;
    for p in rgba.pixels() {
        total += 1;
        if p[2] < 128 {
            low_blue += 1;
        }
    }
    if total == 0 {
        return Err("input image has zero pixels".to_string());
    }
    let frac = low_blue as f64 / total as f64;
    if frac > MAX_LOW_BLUE_FRACTION {
        return Err(format!(
            "does not look like a tangent-space normal map: {:.0}% of pixels have a blue (Z) \
             channel below the midpoint (128), but a valid tangent-space normal points mostly \
             outward (nz >= 0, i.e. blue >= 128) almost everywhere; pass a real normal map, or \
             drop --normal-to-height",
            frac * 100.0
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Overwrite protection + the emitted `.lvl` block
// ---------------------------------------------------------------------------

/// `input` is exempt: with the DEFAULT `--out`/`--name` (input's own directory,
/// input's own stem), converting a `.png` in place resolves the output path to
/// the SAME file as the input — that is a deliberate "shrink this asset in
/// place" request, not a clobber of some unrelated prior output, so it needs no
/// `--force`. Any OTHER pre-existing file still requires it.
fn check_overwrite(paths: &[PathBuf], input: &Path, force: bool) -> Result<(), String> {
    if force {
        return Ok(());
    }
    let input_canon = fs::canonicalize(input).ok();
    let existing: Vec<String> = paths
        .iter()
        .filter(|p| p.exists())
        .filter(
            |p| match (input_canon.as_deref(), fs::canonicalize(p).ok()) {
                (Some(i), Some(o)) => i != o,
                _ => true,
            },
        )
        .map(|p| p.display().to_string())
        .collect();
    if existing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "refusing to overwrite existing file(s) without --force: {}",
        existing.join(", ")
    ))
}

/// Print the `[texture NAME]` block. Only `file` / `max_size` / `height` are
/// ever written — the exact set `re_level` parses (`docs/MATERIALS.md`).
fn emit_block(name: &str, file_path: &Path, max_size: Option<u32>, height_path: Option<&Path>) {
    println!();
    println!("[texture {name}]");
    println!("{:<8} = {}", "file", display_path(file_path));
    if let Some(ms) = max_size {
        println!("{:<8} = {ms}", "max_size");
    }
    if let Some(hp) = height_path {
        println!("{:<8} = {}", "height", display_path(hp));
    }
}

/// The path AS IT WILL BE AUTHORED: relative to the current working directory
/// (the repo root, for the normal `cargo run` / installed-tool invocation) with
/// forward slashes, matching every path already written in `levels/*.lvl`. Falls
/// back to the raw output path (still slash-normalised) when no relative form
/// exists (e.g. the output landed on a different drive than the CWD).
fn display_path(p: &Path) -> String {
    let cwd = std::env::current_dir().ok();
    let rel = cwd.as_deref().and_then(|c| relativize(c, p));
    let chosen = rel.unwrap_or_else(|| p.to_path_buf());
    chosen.to_string_lossy().replace('\\', "/")
}

/// `target` expressed relative to `base`, via canonicalized-component diffing.
/// `None` when the two share no common prefix (e.g. different Windows drives).
fn relativize(base: &Path, target: &Path) -> Option<PathBuf> {
    let base = fs::canonicalize(base).ok()?;
    let target = fs::canonicalize(target).ok()?;
    let base_comps: Vec<_> = base.components().collect();
    let target_comps: Vec<_> = target.components().collect();
    let common = base_comps
        .iter()
        .zip(target_comps.iter())
        .take_while(|(a, b)| a == b)
        .count();
    if common == 0 {
        return None;
    }
    let up = base_comps.len() - common;
    let mut out = PathBuf::new();
    for _ in 0..up {
        out.push("..");
    }
    for c in &target_comps[common..] {
        out.push(c.as_os_str());
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Unit tests — pure-function math, no filesystem I/O beyond a scratch dir
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_longest_edge_preserves_aspect_and_obeys_the_cap() {
        let img = DynamicImage::ImageRgba8(RgbaImage::new(800, 400));
        let capped = cap_longest_edge(img, Some(100));
        assert_eq!(capped.width(), 100);
        assert_eq!(capped.height(), 50, "2:1 aspect preserved");
    }

    #[test]
    fn cap_longest_edge_is_a_noop_when_already_within_the_cap() {
        let img = DynamicImage::ImageRgba8(RgbaImage::new(50, 30));
        let capped = cap_longest_edge(img, Some(100));
        assert_eq!((capped.width(), capped.height()), (50, 30));
    }

    #[test]
    fn cap_longest_edge_is_a_noop_when_max_size_is_none() {
        let img = DynamicImage::ImageRgba8(RgbaImage::new(4000, 10));
        let capped = cap_longest_edge(img, None);
        assert_eq!((capped.width(), capped.height()), (4000, 10));
    }

    #[test]
    fn flat_normal_map_yields_a_constant_height() {
        // Every texel points straight up: R=128 (nx=0), G=128 (ny=0), B=255 (nz=1).
        let mut img = RgbaImage::new(4, 4);
        for p in img.pixels_mut() {
            *p = image::Rgba([128, 128, 255, 255]);
        }
        let heights = normal_to_height(&img);
        assert!(
            heights.iter().all(|&v| v == heights[0]),
            "flat map -> constant height: {heights:?}"
        );
    }

    #[test]
    fn parse_args_rejects_zero_max_size() {
        let err = parse_args(&[
            "x.png".to_string(),
            "--max-size".to_string(),
            "0".to_string(),
        ])
        .unwrap_err();
        assert!(err.contains("max-size"), "{err}");
    }

    #[test]
    fn parse_args_requires_input() {
        assert!(parse_args(&[]).is_err());
    }

    #[test]
    fn parse_args_reads_all_flags() {
        let a = parse_args(&[
            "in.png".to_string(),
            "--out".to_string(),
            "outdir".to_string(),
            "--name".to_string(),
            "n".to_string(),
            "--max-size".to_string(),
            "64".to_string(),
            "--normal-to-height".to_string(),
            "--frames".to_string(),
            "--force".to_string(),
        ])
        .unwrap();
        assert_eq!(a.input, PathBuf::from("in.png"));
        assert_eq!(a.out, Some(PathBuf::from("outdir")));
        assert_eq!(a.name.as_deref(), Some("n"));
        assert_eq!(a.max_size, Some(64));
        assert!(a.normal_to_height && a.frames && a.force);
    }

    #[test]
    fn validate_normal_map_rejects_grayscale() {
        let img = DynamicImage::ImageLuma8(image::GrayImage::new(4, 4));
        assert!(validate_normal_map(&img).is_err());
    }

    #[test]
    fn validate_normal_map_accepts_a_mostly_up_facing_field() {
        let mut img = RgbaImage::new(4, 4);
        for p in img.pixels_mut() {
            *p = image::Rgba([130, 120, 240, 255]);
        }
        assert!(validate_normal_map(&DynamicImage::ImageRgba8(img)).is_ok());
    }

    #[test]
    fn check_overwrite_blocks_without_force_and_allows_with_force() {
        let dir = std::env::temp_dir().join(format!("tex_import_ut_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("exists.png");
        let unrelated_input = dir.join("input.png");
        fs::write(&p, b"x").unwrap();
        fs::write(&unrelated_input, b"y").unwrap();
        assert!(check_overwrite(std::slice::from_ref(&p), &unrelated_input, false).is_err());
        assert!(check_overwrite(std::slice::from_ref(&p), &unrelated_input, true).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_overwrite_exempts_the_input_itself() {
        let dir = std::env::temp_dir().join(format!("tex_import_ut_self_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("in_place.png");
        fs::write(&p, b"x").unwrap();
        // The output path IS the input path (a default-args in-place convert) —
        // must be allowed even without --force.
        assert!(check_overwrite(std::slice::from_ref(&p), &p, false).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }
}
