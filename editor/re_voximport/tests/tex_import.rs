//! Integration tests for the `tex_import` CLI binary.
//!
//! These shell out to the ACTUAL compiled binary via `CARGO_BIN_EXE_tex_import`
//! (the standard cargo mechanism for testing a package's own `[[bin]]` targets),
//! rather than calling private functions, since `tex_import`'s own logic lives
//! entirely in `src/bin/tex_import.rs` (a binary target has no public library
//! surface for an integration test to import).
//!
//! Fixtures used:
//! - `assets/texture/gen_anim.gif` — a small, repo-tracked, GENERATED 48x24 GIF
//!   (`re_voximport::texfixture::anim_gif`, replacing a previous fixture that
//!   was an untracked download of unknown origin). Its frame count is verified
//!   INDEPENDENTLY here (via `image::codecs::gif`), not assumed, so the test still
//!   means something if the source asset ever changes.
//! - Synthetic images built in-memory (a Gaussian bump normal map, a plain RGBA
//!   gradient) so the resize/normal-to-height assertions do not depend on any
//!   external file.

use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, Rgba, RgbaImage};
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

const fn bin_path() -> &'static str {
    env!("CARGO_BIN_EXE_tex_import")
}

/// A fresh, uniquely-named scratch directory under the OS temp dir — never under
/// `assets/` or `levels/` — removed and recreated so each test starts clean.
/// A fresh scratch dir under the workspace's OWN `target/` (never `assets/` or
/// `levels/`, but still on the SAME volume as the repo). On Windows the OS temp
/// dir can sit on a different drive letter than the repo (`C:` vs. a `D:`
/// checkout), which makes `tex_import`'s "print a repo-relative path" behaviour
/// legitimately fall back to an absolute path (there is no relative form across
/// drives) — using a same-volume scratch dir here keeps that assertion
/// meaningful instead of exercising the cross-drive fallback by accident.
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = repo_root()
        .join("target")
        .join("tex_import_it_scratch")
        .join(format!("{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn run_tex_import(args: &[&str]) -> Run {
    let out = Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn tex_import: {e}"));
    Run {
        status: out.status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn write_png(path: &Path, img: &RgbaImage) {
    img.save(path)
        .unwrap_or_else(|e| panic!("write fixture {}: {e}", path.display()));
}

// ---------------------------------------------------------------------------
// 1. Resize: honours the cap, preserves aspect, reports before/after
// ---------------------------------------------------------------------------

#[test]
fn resize_honours_the_cap_and_preserves_aspect() {
    let dir = scratch_dir("resize");
    let src = dir.join("wide.png");
    // 2:1 aspect, comfortably over any sane --max-size.
    let img = RgbaImage::from_fn(800, 400, |x, y| {
        Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255])
    });
    write_png(&src, &img);

    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "wide_capped",
        "--max-size",
        "100",
    ]);
    assert!(
        run.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );

    let out_path = dir.join("wide_capped.png");
    assert!(
        out_path.is_file(),
        "resized copy must be written: {}",
        out_path.display()
    );
    let decoded = image::open(&out_path).expect("decode resized output");
    assert_eq!(decoded.width(), 100, "longest edge capped to 100");
    assert_eq!(decoded.height(), 50, "2:1 aspect preserved");

    // The report line must state the real before/after dims and an RGBA byte cost.
    assert!(
        run.stdout.contains("800x400"),
        "before dims reported:\n{}",
        run.stdout
    );
    assert!(
        run.stdout.contains("100x50"),
        "after dims reported:\n{}",
        run.stdout
    );
    let before_bytes = 800u64 * 400 * 4;
    let after_bytes = 100u64 * 50 * 4;
    assert!(
        run.stdout.contains(&before_bytes.to_string()),
        "before RGBA byte cost reported:\n{}",
        run.stdout
    );
    assert!(
        run.stdout.contains(&after_bytes.to_string()),
        "after RGBA byte cost reported:\n{}",
        run.stdout
    );

    // The emitted block carries the cap forward as `max_size`.
    assert!(
        run.stdout.contains("[texture wide_capped]"),
        "{}",
        run.stdout
    );
    assert!(run.stdout.contains("max_size = 100"), "{}", run.stdout);
}

#[test]
fn resize_with_no_max_size_leaves_dimensions_untouched() {
    let dir = scratch_dir("nocap");
    let src = dir.join("small.png");
    let img = RgbaImage::from_pixel(37, 21, Rgba([10, 20, 30, 255]));
    write_png(&src, &img);

    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "small",
    ]);
    assert!(run.status.success(), "stderr:\n{}", run.stderr);

    let decoded = image::open(dir.join("small.png")).expect("decode output");
    assert_eq!((decoded.width(), decoded.height()), (37, 21));
    // No --max-size was given, so the block must not claim one.
    assert!(!run.stdout.contains("max_size"), "{}", run.stdout);
}

// ---------------------------------------------------------------------------
// 2. normal -> height: recovers a synthetic bump (real numeric assertion)
// ---------------------------------------------------------------------------

/// Builds a smooth Gaussian-bump height field analytically, derives the EXACT
/// tangent-space normal map that height field implies (`n = normalize(-dh/dx,
/// -dh/dy, 1)`), and returns both the encoded RGBA normal map and the analytic
/// height field it came from (for comparison against the tool's RECOVERED
/// height map — this validates the actual gradient-integration math, not just
/// "the tool ran").
#[allow(clippy::similar_names)] // gradient_x / gradient_y ARE the two distinct axes
fn synthetic_bump(w: u32, h: u32) -> (RgbaImage, Vec<f64>) {
    let cx = f64::from(w) / 2.0;
    let cy = f64::from(h) / 2.0;
    let sigma = f64::from(w.min(h)) / 6.0;
    let sigma_sq = sigma * sigma;
    let amp = 40.0_f64;

    let h0 = |x: f64, y: f64| -> f64 {
        let (dx, dy) = (x - cx, y - cy);
        amp * (-(dx * dx + dy * dy) / (2.0 * sigma_sq)).exp()
    };
    let gradient_x = |x: f64, y: f64| -> f64 {
        let dx = x - cx;
        h0(x, y) * (-dx / sigma_sq)
    };
    let gradient_y = |x: f64, y: f64| -> f64 {
        let dy = y - cy;
        h0(x, y) * (-dy / sigma_sq)
    };

    let mut img = RgbaImage::new(w, h);
    let mut heights = vec![0.0_f64; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (f64::from(x), f64::from(y));
            let gx = gradient_x(xf, yf);
            let gy = gradient_y(xf, yf);
            let nz = 1.0 / (gx * gx + gy * gy + 1.0).sqrt();
            let nx = -gx * nz;
            let ny = -gy * nz;
            let enc =
                |n: f64| -> u8 { (f64::midpoint(n, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8 };
            img.put_pixel(x, y, Rgba([enc(nx), enc(ny), enc(nz), 255]));
            heights[(y * w + x) as usize] = h0(xf, yf);
        }
    }
    (img, heights)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let (mut cov, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

#[test]
fn normal_to_height_recovers_a_synthetic_bump() {
    let dir = scratch_dir("bump");
    let (normal_map, analytic_heights) = synthetic_bump(48, 48);
    let src = dir.join("bump_normal.png");
    write_png(&src, &normal_map);

    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "bump",
        "--normal-to-height",
    ]);
    assert!(
        run.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );

    let height_path = dir.join("bump_height.png");
    assert!(
        height_path.is_file(),
        "height map must be written: {}",
        height_path.display()
    );
    let recovered = image::open(&height_path)
        .expect("decode height map")
        .to_luma8();
    assert_eq!((recovered.width(), recovered.height()), (48, 48));

    let recovered_f: Vec<f64> = recovered.pixels().map(|p| f64::from(p.0[0])).collect();
    let corr = pearson(&analytic_heights, &recovered_f);
    assert!(
        corr > 0.85,
        "recovered height must correlate strongly with the analytic bump it came from; \
         got pearson r = {corr}"
    );

    // The centre (peak of the bump) must be recovered as one of the HIGHEST
    // points, not the lowest — a sign-flipped integration would fail this.
    let center = recovered.get_pixel(24, 24).0[0];
    let corner = recovered.get_pixel(1, 1).0[0];
    assert!(
        center > corner,
        "bump peak ({center}) must read higher than a flat corner ({corner})"
    );

    // A colour copy is also written (so the block below is self-contained) and
    // the block carries `height =`.
    assert!(
        dir.join("bump.png").is_file(),
        "colour copy must accompany the height map"
    );
    assert!(run.stdout.contains("[texture bump]"), "{}", run.stdout);
    assert!(run.stdout.contains("height   = "), "{}", run.stdout);
    assert!(run.stdout.contains("bump_height.png"), "{}", run.stdout);
}

#[test]
fn normal_to_height_rejects_a_photo_like_input() {
    let dir = scratch_dir("notnormal");
    let src = dir.join("photo.png");
    // A checkerboard with a low, wildly varying blue channel: nothing like a
    // tangent-space normal map's blue-dominant Z channel.
    let img = RgbaImage::from_fn(16, 16, |x, y| {
        let v = if (x + y) % 2 == 0 { 10 } else { 5 };
        Rgba([v, v, v, 255])
    });
    write_png(&src, &img);

    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "notnormal",
        "--normal-to-height",
    ]);
    assert!(
        !run.status.success(),
        "must exit non-zero on a non-normal-looking image"
    );
    assert!(!run.stderr.is_empty(), "must explain why");
    assert!(
        !dir.join("notnormal_height.png").exists(),
        "must not write a bogus height map"
    );
}

// ---------------------------------------------------------------------------
// 3. GIF -> frames
// ---------------------------------------------------------------------------

fn decode_gif_frame_count(path: &Path) -> usize {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let decoder = GifDecoder::new(std::io::BufReader::new(f))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    decoder
        .into_frames()
        .collect_frames()
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .len()
}

#[test]
fn frames_flag_reports_and_preserves_every_gif_frame() {
    let src = repo_root()
        .join("assets")
        .join("texture")
        .join("gen_anim.gif");
    assert!(src.is_file(), "fixture must ship: {}", src.display());

    // Ground truth, decoded independently of the tool.
    let expected_frames = decode_gif_frame_count(&src);
    assert!(
        expected_frames > 1,
        "fixture must actually be animated: {expected_frames} frames"
    );

    let dir = scratch_dir("gif");
    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "clip",
        "--frames",
        "--max-size",
        "128",
    ]);
    assert!(
        run.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );

    assert!(
        run.stdout.contains(&format!("{expected_frames} frames")),
        "frame count reported (expected {expected_frames}):\n{}",
        run.stdout
    );
    // The report must also state a frame rate, not just a count.
    assert!(run.stdout.to_lowercase().contains("fps"), "{}", run.stdout);

    let out_gif = dir.join("clip.gif");
    assert!(
        out_gif.is_file(),
        "GIF output must be written (never exploded into PNGs)"
    );
    // Not exploded into N PNGs alongside it.
    let png_siblings: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
        .collect();
    assert!(
        png_siblings.is_empty(),
        "must not explode a GIF into PNG frames: {png_siblings:?}"
    );

    let got_frames = decode_gif_frame_count(&out_gif);
    assert_eq!(
        got_frames, expected_frames,
        "every frame of the source GIF must survive"
    );

    assert!(run.stdout.contains("[texture clip]"), "{}", run.stdout);
    assert!(run.stdout.contains("file     = "), "{}", run.stdout);
    assert!(run.stdout.contains("clip.gif"), "{}", run.stdout);
}

#[test]
fn without_frames_flag_a_gif_converts_only_its_first_frame() {
    let src = repo_root()
        .join("assets")
        .join("texture")
        .join("gen_anim.gif");
    assert!(src.is_file());

    let dir = scratch_dir("gif_nofr");
    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "clip1",
    ]);
    assert!(run.status.success(), "stderr:\n{}", run.stderr);

    // Without --frames the primary output is a static PNG (first frame only) —
    // matches `re_level`'s own `[texture] file = x.gif` behaviour today.
    assert!(
        dir.join("clip1.png").is_file(),
        "static single-frame conversion"
    );
    assert!(!dir.join("clip1.gif").exists());
}

// ---------------------------------------------------------------------------
// 4. Emitted block parses back through `re_level::load_string`
// ---------------------------------------------------------------------------

/// Mirrors the minimal `.lvl` skeleton `re_voximport`'s own tests use
/// (`cells_shell` in `src/tests.rs`) so a bare `[texture]` section is the only
/// thing under test.
fn minimal_level(body: &str) -> String {
    format!(
        "[meta]\nname = cells-test\n[room]\nsize = 3 2 2\nambient = solid\n\
         [pedestal]\ncenter = 1 1 1\nvolume = 1 1 1\n[door]\npos = 2 1 1\n\
         [player]\nspawn = 1 1 1\n{body}"
    )
}

fn extract_texture_block(stdout: &str) -> &str {
    let start = stdout
        .find("[texture ")
        .expect("a [texture ...] block must be printed");
    let rest = &stdout[start..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    rest[..end].trim_end()
}

#[test]
fn emitted_block_parses_through_re_level_load_string() {
    let dir = scratch_dir("parseback");
    let src = dir.join("mat.png");
    write_png(
        &src,
        &RgbaImage::from_pixel(8, 8, Rgba([200, 100, 50, 255])),
    );

    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "mat",
        "--max-size",
        "64",
    ]);
    assert!(run.status.success(), "stderr:\n{}", run.stderr);

    let block = extract_texture_block(&run.stdout);
    let level_text = minimal_level(block);
    let level = re_level::load_string(&level_text, "<tex_import-parseback>")
        .unwrap_or_else(|e| panic!("emitted block must parse: {e}\n\nblock was:\n{block}"));

    assert_eq!(
        level.textures.len(),
        1,
        "one [texture] parsed from the emitted block"
    );
    assert_eq!(level.textures[0].name, "mat");
    assert!(
        !level.textures[0].file.is_empty(),
        "file key must be present and non-empty"
    );
    assert_eq!(
        level.textures[0].max_size,
        Some(64),
        "max_size key must round-trip"
    );

    // The path must be authored relative (never an absolute temp path baked
    // into the pasted block).
    assert!(
        !Path::new(&level.textures[0].file).is_absolute(),
        "emitted file path must be relative, not absolute: {}",
        level.textures[0].file
    );
}

// ---------------------------------------------------------------------------
// 5. Overwrite protection
// ---------------------------------------------------------------------------

#[test]
fn refuses_to_overwrite_without_force_and_allows_with_force() {
    let dir = scratch_dir("overwrite");
    // NAME deliberately differs from the source's own stem, so the output path
    // is a genuinely SEPARATE file from the input (not the "convert in place"
    // exemption — see `emitted_block_parses_through_re_level_load_string` /
    // `resize_with_no_max_size_leaves_dimensions_untouched` for that case).
    let src = dir.join("source.png");
    write_png(&src, &RgbaImage::from_pixel(5, 5, Rgba([1, 2, 3, 255])));

    let first = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "derived",
    ]);
    assert!(first.status.success(), "stderr:\n{}", first.stderr);
    let out_path = dir.join("derived.png");
    assert!(out_path.is_file());
    let mtime_before = std::fs::metadata(&out_path).unwrap().modified().unwrap();

    let second = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "derived",
    ]);
    assert!(
        !second.status.success(),
        "must refuse to overwrite without --force"
    );
    assert!(
        second.stderr.to_lowercase().contains("overwrite") || second.stderr.contains("--force"),
        "stderr must explain why: {}",
        second.stderr
    );
    let mtime_after_refused = std::fs::metadata(&out_path).unwrap().modified().unwrap();
    assert_eq!(
        mtime_before, mtime_after_refused,
        "the refused run must not touch the file"
    );

    let third = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--name",
        "derived",
        "--force",
    ]);
    assert!(
        third.status.success(),
        "--force must allow the overwrite: {}",
        third.stderr
    );
}

// ---------------------------------------------------------------------------
// Bad input: clean errors, never a panic
// ---------------------------------------------------------------------------

#[test]
fn missing_input_file_is_a_clean_error_not_a_panic() {
    let dir = scratch_dir("missing");
    let run = run_tex_import(&[
        dir.join("does_not_exist.png").to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
    ]);
    assert!(!run.status.success());
    assert!(!run.stderr.is_empty());
    assert!(
        !run.stderr.to_lowercase().contains("panicked"),
        "must not panic: {}",
        run.stderr
    );
}

#[test]
fn max_size_zero_is_a_clean_error_not_a_panic() {
    let dir = scratch_dir("zerocap");
    let src = dir.join("x.png");
    write_png(&src, &RgbaImage::from_pixel(4, 4, Rgba([9, 9, 9, 255])));
    let run = run_tex_import(&[
        src.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--max-size",
        "0",
    ]);
    assert!(!run.status.success());
    assert!(
        !run.stderr.to_lowercase().contains("panicked"),
        "must not panic: {}",
        run.stderr
    );
}

#[test]
fn undecodable_input_is_a_clean_error_not_a_panic() {
    let dir = scratch_dir("garbage");
    let src = dir.join("garbage.png");
    std::fs::write(&src, b"this is not an image file, just bytes").unwrap();
    let run = run_tex_import(&[src.to_str().unwrap(), "--out", dir.to_str().unwrap()]);
    assert!(!run.status.success());
    assert!(!run.stderr.is_empty());
    assert!(
        !run.stderr.to_lowercase().contains("panicked"),
        "must not panic: {}",
        run.stderr
    );
}
