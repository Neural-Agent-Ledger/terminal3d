//! Dev-only gallery dumper — NOT part of the shipped build (this whole file
//! is `#[cfg(test)]`, see `lib.rs`'s `mod gallery;` declaration). Renders
//! every effect at 2 seeds × 4 time steps into ANSI truecolor dumps plus one
//! contact-sheet dump, for a human to look at directly — a visual review aid,
//! not an automated check.
//!
//! Run with:
//! `cargo test -p re_asciifx --lib gallery -- --ignored --nocapture`
//!
//! Output lands in the directory named by the `ASCII_FX_GALLERY_DIR` env var,
//! falling back to `.` so a stray `cargo test` run without the env var still
//! produces SOMETHING rather than failing to find a directory. A companion
//! Python script
//! (`render_gallery.py`, written to the same scratch dir, NOT part of this
//! crate) converts each `.ans` to a `.png` — see this module's own
//! `write_render_script`.

use super::circuit::Circuit;
use super::dna::Dna;
use super::eye::{Eye, EyeVariant};
use super::eye_swarm::EyeSwarm;
use super::heartbeat::Heartbeat;
use super::neural_net::NeuralNet;
use super::oscilloscope::Oscilloscope;
use super::plasma::Plasma;
use super::radar::Radar;
use super::seeds::Seeds;
use super::tunnel::Tunnel;
use super::{put_str_clipped, AsciiEffect, Rect};
use re_core::Framebuffer;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

const TILE_COLS: usize = 70;
const TILE_ROWS: usize = 26;
const GRID_COLS: usize = 4;
const GRID_ROWS: usize = 2;
const LABEL_COLOR: u32 = 0x00FF_D080;

/// The four time steps every per-effect detail sheet samples.
const TIME_STEPS_MS: [u64; 4] = [200, 1000, 2400, 4500];
/// The two seeds every per-effect detail sheet samples.
const SEEDS: [u64; 2] = [1, 2];

fn out_dir() -> PathBuf {
    std::env::var("ASCII_FX_GALLERY_DIR").map_or_else(|_| PathBuf::from("."), PathBuf::from)
}

/// Write `fb` as a `==== TRUECOLOR GRID <cols>x<rows> ====` ANSI dump — every
/// cell emits both an explicit `38;2;r;g;b` (fg) and `48;2;r;g;b` (bg) SGR
/// sequence before its glyph, so a reader (this crate's own Python
/// companion, or any truecolor-aware terminal) never has to track state
/// across a row boundary.
fn write_ansi_dump(fb: &Framebuffer, path: &Path) {
    let mut out = String::new();
    let _ = writeln!(out, "==== TRUECOLOR GRID {}x{} ====", fb.cols(), fb.rows());
    for row in 0..fb.rows() {
        for col in 0..fb.cols() {
            let cell = fb.get(col, row).copied().unwrap_or_default();
            let (fr, fg_, fb_) = unpack(cell.fg);
            let (br, bgg, bb) = unpack(cell.bg);
            let glyph = char::from_u32(cell.glyph).unwrap_or(' ');
            let _ = write!(
                out,
                "\x1b[38;2;{fr};{fg_};{fb_}m\x1b[48;2;{br};{bgg};{bb}m{glyph}"
            );
        }
        out.push_str("\x1b[0m\n");
    }
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("create {path:?}: {e}"));
    f.write_all(out.as_bytes())
        .unwrap_or_else(|e| panic!("write {path:?}: {e}"));
}

fn unpack(c: u32) -> (u8, u8, u8) {
    (
        ((c >> 16) & 0xFF) as u8,
        ((c >> 8) & 0xFF) as u8,
        (c & 0xFF) as u8,
    )
}

fn tile_rect(col_idx: usize, row_idx: usize) -> (Rect, Rect) {
    let col0 = col_idx * TILE_COLS;
    let row0 = row_idx * TILE_ROWS;
    let label_rect = Rect::new(col0, row0, TILE_COLS, 1);
    let body_rect = Rect::new(col0, row0 + 1, TILE_COLS, TILE_ROWS - 1);
    (label_rect, body_rect)
}

/// One effect's detail sheet: `SEEDS.len()` rows × `TIME_STEPS_MS.len()`
/// columns, each tile labelled `seed=<n> t=<ms>ms`.
fn dump_effect_sheet<F>(name: &str, mut build: F)
where
    F: FnMut(u64, Rect) -> Box<dyn AsciiEffect>,
{
    let cols = TILE_COLS * TIME_STEPS_MS.len();
    let rows = TILE_ROWS * SEEDS.len();
    let mut fb = Framebuffer::new(cols, rows);
    for (row_idx, &seed) in SEEDS.iter().enumerate() {
        for (col_idx, &ms) in TIME_STEPS_MS.iter().enumerate() {
            let (label_rect, body_rect) = tile_rect(col_idx, row_idx);
            let effect = build(seed, body_rect);
            effect.paint(&mut fb, body_rect, Duration::from_millis(ms));
            let label = format!("{name} seed={seed} t={ms}ms");
            put_str_clipped(
                &mut fb,
                label_rect,
                label_rect.col0 as i64,
                label_rect.row0 as i64,
                &label,
                LABEL_COLOR,
                0x0000_0000,
            );
        }
    }
    let dir = out_dir();
    write_ansi_dump(&fb, &dir.join(format!("{name}.ans")));
}

#[test]
#[ignore = "dev-only gallery dumper; run explicitly with --ignored --nocapture"]
fn dump_all_effect_sheets() {
    dump_effect_sheet("eye", |seed, rect| Box::new(Eye::random(seed, rect)));
    dump_effect_sheet("eye_swarm", |seed, rect| {
        Box::new(EyeSwarm::random_with_count(seed, rect, 6))
    });
    dump_effect_sheet("circuit", |seed, rect| {
        Box::new(Circuit::random(seed, rect))
    });
    dump_effect_sheet("neural_net", |seed, rect| {
        Box::new(NeuralNet::random(seed, rect))
    });
    dump_effect_sheet("seeds", |seed, rect| Box::new(Seeds::random(seed, rect)));
    dump_effect_sheet("oscilloscope", |seed, rect| {
        Box::new(Oscilloscope::random(seed, rect))
    });
    dump_effect_sheet("radar", |seed, rect| Box::new(Radar::random(seed, rect)));
    dump_effect_sheet("heartbeat", |seed, rect| {
        Box::new(Heartbeat::random(seed, rect))
    });

    // Bonus: the eye's 5 variants side by side (not one of the 8 required
    // sheets — the eye is the splash-replacement candidate and earns extra
    // scrutiny). 5 tiles fit a 1-row-of-5 layout at a wider tile.
    let variant_cols = 66usize;
    let variant_rows = 30usize;
    let mut fb = Framebuffer::new(variant_cols * 5, variant_rows);
    for (i, variant) in [
        EyeVariant::Human,
        EyeVariant::CatSlit,
        EyeVariant::Reptile,
        EyeVariant::Cyber,
        EyeVariant::Insect,
    ]
    .into_iter()
    .enumerate()
    {
        let col0 = i * variant_cols;
        let label_rect = Rect::new(col0, 0, variant_cols, 1);
        let body_rect = Rect::new(col0, 1, variant_cols, variant_rows - 1);
        // Gaze pinned dead-centre: this sheet exists to show off each
        // variant's PUPIL SHAPE cleanly, and an autonomous saccade can (very
        // correctly — real eyes hood their pupil when gazing off-centre) land
        // anywhere at a sampled `t`, which is the wrong moment for a shape
        // showcase specifically.
        let eye = Eye::random(10 + i as u64, body_rect)
            .with_variant(variant)
            .with_target(Some(body_rect.center()));
        eye.paint(&mut fb, body_rect, Duration::from_millis(600));
        put_str_clipped(
            &mut fb,
            label_rect,
            label_rect.col0 as i64,
            0,
            &format!("{variant:?}"),
            LABEL_COLOR,
            0x0000_0000,
        );
    }
    write_ansi_dump(&fb, &out_dir().join("eye_variants.ans"));

    // The main contact sheet: one representative frame per effect, 4x2.
    //
    // Every effect must be BUILT with the exact same `Rect` it is later
    // PAINTED with. `Circuit`/`NeuralNet`/`EyeSwarm` bake absolute cell
    // coordinates (chip corners, node positions …) into their `random()`
    // constructor from the rect they are given there — unlike `Eye`, which
    // re-derives its geometry from whatever rect `paint()` receives (see
    // `Eye::random`'s doc: "`rect` is not read at all here"). Building one of
    // the position-baking effects against a throwaway `Rect::new(0, 0, …)`
    // and then painting it into an offset tile rect made every one of them
    // paint into the WRONG absolute region — invisible on this canvas, since
    // `paint()` clips to the rect it is CALLED with, not the one the effect
    // was built with — and that is exactly what the first gallery render
    // (2026-09-24) showed: `circuit`/`neural_net`/`eye_swarm` came out
    // entirely blank while `eye`/`seeds`/`oscilloscope`/`radar`/`heartbeat`
    // (all rect-independent at construction) did not. Building each effect
    // AFTER computing its tile's `body_rect`, from that same `body_rect`,
    // fixes all three at once.
    let cols = TILE_COLS * GRID_COLS;
    let rows = TILE_ROWS * GRID_ROWS;
    let mut fb = Framebuffer::new(cols, rows);
    let t = Duration::from_millis(900);
    let names = [
        "eye",
        "eye_swarm",
        "circuit",
        "neural_net",
        "seeds",
        "oscilloscope",
        "radar",
        "heartbeat",
    ];
    for (idx, name) in names.iter().enumerate() {
        let col_idx = idx % GRID_COLS;
        let row_idx = idx / GRID_COLS;
        let (label_rect, body_rect) = tile_rect(col_idx, row_idx);
        let effect: Box<dyn AsciiEffect> = match *name {
            "eye" => Box::new(Eye::random(7, body_rect)),
            "eye_swarm" => Box::new(EyeSwarm::random_with_count(7, body_rect, 6)),
            "circuit" => Box::new(Circuit::random(7, body_rect)),
            "neural_net" => Box::new(NeuralNet::random(7, body_rect)),
            "seeds" => Box::new(Seeds::random(7, body_rect)),
            "oscilloscope" => Box::new(Oscilloscope::random(7, body_rect)),
            "radar" => Box::new(Radar::random(7, body_rect)),
            _ => Box::new(Heartbeat::random(7, body_rect)),
        };
        effect.paint(&mut fb, body_rect, t);
        put_str_clipped(
            &mut fb,
            label_rect,
            label_rect.col0 as i64,
            label_rect.row0 as i64,
            name,
            LABEL_COLOR,
            0x0000_0000,
        );
    }
    write_ansi_dump(&fb, &out_dir().join("gallery.ans"));

    write_render_script(&out_dir());
}

/// `gallery2` samples 2 seeds × 2 times per effect, unlike `gallery`'s single
/// representative frame per effect — so each effect's block below is itself
/// a small 2(seed)×2(time) grid, and the 8 effects are laid out in the SAME
/// 4-column × 2-row arrangement `dump_all_effect_sheets`'s contact sheet
/// uses, just with each cell twice as wide/tall to fit its own sub-grid —
/// directly comparable to the original `gallery.png` at a glance.
const GALLERY2_SEEDS: [u64; 2] = [3, 4];
const GALLERY2_TIME_STEPS_MS: [u64; 2] = [500, 3200];

fn build_effect_block(name: &str) -> Framebuffer {
    let cols = TILE_COLS * GALLERY2_TIME_STEPS_MS.len();
    let rows = TILE_ROWS * GALLERY2_SEEDS.len();
    let mut fb = Framebuffer::new(cols, rows);
    for (row_idx, &seed) in GALLERY2_SEEDS.iter().enumerate() {
        for (col_idx, &ms) in GALLERY2_TIME_STEPS_MS.iter().enumerate() {
            let col0 = col_idx * TILE_COLS;
            let row0 = row_idx * TILE_ROWS;
            let label_rect = Rect::new(col0, row0, TILE_COLS, 1);
            let body_rect = Rect::new(col0, row0 + 1, TILE_COLS, TILE_ROWS - 1);
            let effect: Box<dyn AsciiEffect> = match name {
                "eye" => Box::new(Eye::random(seed, body_rect)),
                "eye_swarm" => Box::new(EyeSwarm::random(seed, body_rect)),
                "circuit" => Box::new(Circuit::random(seed, body_rect)),
                "neural_net" => Box::new(NeuralNet::random(seed, body_rect)),
                "seeds" => Box::new(Seeds::random(seed, body_rect)),
                "oscilloscope" => Box::new(Oscilloscope::random(seed, body_rect)),
                "radar" => Box::new(Radar::random(seed, body_rect)),
                _ => Box::new(Heartbeat::random(seed, body_rect)),
            };
            effect.paint(&mut fb, body_rect, Duration::from_millis(ms));
            let label = format!("{name} seed={seed} t={ms}ms");
            put_str_clipped(
                &mut fb,
                label_rect,
                col0 as i64,
                row0 as i64,
                &label,
                LABEL_COLOR,
                0x0000_0000,
            );
        }
    }
    fb
}

/// Blit `src` into `dst` at `(col0, row0)` — every cell, no clipping logic
/// needed since callers always size `dst` to exactly fit every block.
fn blit_block(dst: &mut Framebuffer, src: &Framebuffer, col0: usize, row0: usize) {
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            dst.set(
                col0 + col,
                row0 + row,
                src.get(col, row).copied().unwrap_or_default(),
            );
        }
    }
}

#[test]
#[ignore = "dev-only gallery dumper; run explicitly with --ignored --nocapture"]
fn dump_gallery2() {
    let names = [
        "eye",
        "eye_swarm",
        "circuit",
        "neural_net",
        "seeds",
        "oscilloscope",
        "radar",
        "heartbeat",
    ];
    let block_cols = TILE_COLS * GALLERY2_TIME_STEPS_MS.len();
    let block_rows = TILE_ROWS * GALLERY2_SEEDS.len();
    let cols = block_cols * GRID_COLS;
    let rows = block_rows * GRID_ROWS;
    let mut fb = Framebuffer::new(cols, rows);
    for (idx, name) in names.iter().enumerate() {
        let block = build_effect_block(name);
        let col0 = (idx % GRID_COLS) * block_cols;
        let row0 = (idx / GRID_COLS) * block_rows;
        blit_block(&mut fb, &block, col0, row0);
    }
    write_ansi_dump(&fb, &out_dir().join("gallery2.ans"));

    // eye_variants2: same construction as `eye_variants.ans` — the eye's own
    // geometry has not changed since that sheet was reviewed and approved;
    // this is a fresh, identically-named-for-this-round snapshot alongside
    // gallery2, not a re-test of the eye itself.
    dump_eye_variants("eye_variants2.ans");

    write_render_script(&out_dir());
}

/// One row of all 5 [`EyeVariant`]s side by side, gaze pinned dead-centre
/// (see the inline doc on the original `eye_variants.ans` construction in
/// `dump_all_effect_sheets` — showing off pupil SHAPE cleanly, not wherever
/// an autonomous saccade happens to be looking at a sampled `t`). Shared by
/// every `eye_variantsN.ans` dumper so each round's sheet is directly
/// comparable to the last.
fn dump_eye_variants(filename: &str) {
    let variant_cols = 66usize;
    let variant_rows = 30usize;
    let mut vfb = Framebuffer::new(variant_cols * 5, variant_rows);
    for (i, variant) in [
        EyeVariant::Human,
        EyeVariant::CatSlit,
        EyeVariant::Reptile,
        EyeVariant::Cyber,
        EyeVariant::Insect,
    ]
    .into_iter()
    .enumerate()
    {
        let col0 = i * variant_cols;
        let label_rect = Rect::new(col0, 0, variant_cols, 1);
        let body_rect = Rect::new(col0, 1, variant_cols, variant_rows - 1);
        let eye = Eye::random(10 + i as u64, body_rect)
            .with_variant(variant)
            .with_target(Some(body_rect.center()));
        eye.paint(&mut vfb, body_rect, Duration::from_millis(600));
        put_str_clipped(
            &mut vfb,
            label_rect,
            label_rect.col0 as i64,
            0,
            &format!("{variant:?}"),
            LABEL_COLOR,
            0x0000_0000,
        );
    }
    write_ansi_dump(&vfb, &out_dir().join(filename));
}

#[test]
#[ignore = "dev-only gallery dumper; run explicitly with --ignored --nocapture"]
fn dump_eye_variants3() {
    // A fresh snapshot after the flat-topped-lid-arc fix in `eye.rs`'s
    // `lid_half_open`/`FIT_MARGIN_FRAC`/lash-line, to confirm the arch reads
    // correctly in every variant.
    dump_eye_variants("eye_variants3.ans");
    write_render_script(&out_dir());
}

/// `gallery3`'s effect list: the two weakest effects from the prior round,
/// reworked (`radar`, `seeds`), plus the three brand-new ones (`plasma`,
/// `tunnel`, `dna`) — every effect touched in this pass, and nothing that
/// was untouched (an unchanged `eye`/`circuit`/etc. has no new information
/// to show here; `gallery2.ans` already covers it).
const GALLERY3_NAMES: [&str; 5] = ["radar", "seeds", "plasma", "tunnel", "dna"];
const GALLERY3_GRID_COLS: usize = 3;
const GALLERY3_GRID_ROWS: usize = 2;

fn build_effect_block3(name: &str) -> Framebuffer {
    let cols = TILE_COLS * GALLERY2_TIME_STEPS_MS.len();
    let rows = TILE_ROWS * GALLERY2_SEEDS.len();
    let mut fb = Framebuffer::new(cols, rows);
    for (row_idx, &seed) in GALLERY2_SEEDS.iter().enumerate() {
        for (col_idx, &ms) in GALLERY2_TIME_STEPS_MS.iter().enumerate() {
            let col0 = col_idx * TILE_COLS;
            let row0 = row_idx * TILE_ROWS;
            let label_rect = Rect::new(col0, row0, TILE_COLS, 1);
            let body_rect = Rect::new(col0, row0 + 1, TILE_COLS, TILE_ROWS - 1);
            let effect: Box<dyn AsciiEffect> = match name {
                "radar" => Box::new(Radar::random(seed, body_rect)),
                "seeds" => Box::new(Seeds::random(seed, body_rect)),
                "plasma" => Box::new(Plasma::random(seed, body_rect)),
                "tunnel" => Box::new(Tunnel::random(seed, body_rect)),
                _ => Box::new(Dna::random(seed, body_rect)),
            };
            effect.paint(&mut fb, body_rect, Duration::from_millis(ms));
            let label = format!("{name} seed={seed} t={ms}ms");
            put_str_clipped(
                &mut fb,
                label_rect,
                col0 as i64,
                row0 as i64,
                &label,
                LABEL_COLOR,
                0x0000_0000,
            );
        }
    }
    fb
}

#[test]
#[ignore = "dev-only gallery dumper; run explicitly with --ignored --nocapture"]
fn dump_gallery3() {
    let block_cols = TILE_COLS * GALLERY2_TIME_STEPS_MS.len();
    let block_rows = TILE_ROWS * GALLERY2_SEEDS.len();
    let cols = block_cols * GALLERY3_GRID_COLS;
    let rows = block_rows * GALLERY3_GRID_ROWS;
    let mut fb = Framebuffer::new(cols, rows);
    for (idx, name) in GALLERY3_NAMES.iter().enumerate() {
        let block = build_effect_block3(name);
        let col0 = (idx % GALLERY3_GRID_COLS) * block_cols;
        let row0 = (idx / GALLERY3_GRID_COLS) * block_rows;
        blit_block(&mut fb, &block, col0, row0);
    }
    write_ansi_dump(&fb, &out_dir().join("gallery3.ans"));
    write_render_script(&out_dir());
}

/// A small, self-contained companion script (Python + Pillow) that turns any
/// `.ans` dump this module writes into a `.png` — a purpose-built copy rather
/// than reusing the full-game-dump renderer, because that script hardcodes a
/// 330×85 canvas and this gallery's tile grids are a different shape per
/// sheet; the two are independent tools that happen to share a technique.
fn write_render_script(dir: &Path) {
    let script = r#"import sys, re
from PIL import Image, ImageDraw, ImageFont

ESC = "\x1b"
SPLIT = re.compile("(" + ESC + r"\[[0-9;:]*m)")

def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    marker = txt.find("==== TRUECOLOR GRID")
    header_end = txt.find("\n", marker) + 1
    header = txt[marker:header_end]
    m = re.search(r"(\d+)x(\d+)", header)
    cols, rows = int(m.group(1)), int(m.group(2))
    txt = txt[header_end:]
    fg = bg = (0, 0, 0)
    cells = []
    for tok in SPLIT.split(txt):
        if tok.startswith(ESC):
            p = [int(x) if x.isdigit() else -1 for x in tok[2:-1].split(";") if x]
            k = 0
            while k < len(p):
                if p[k] == 38 and k + 1 < len(p) and p[k + 1] == 2:
                    fg = tuple(p[k + 2:k + 5]); k += 5
                elif p[k] == 48 and k + 1 < len(p) and p[k + 1] == 2:
                    bg = tuple(p[k + 2:k + 5]); k += 5
                else:
                    k += 1
            continue
        for ch in tok:
            if ch in "\r\n":
                continue
            cells.append((fg, bg, ch))
    return cols, rows, cells

def render(path, out, cw=8, ch=16):
    cols, rows, cells = parse(path)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", int(ch * 0.85))
    except Exception:
        font = ImageFont.load_default()
    im = Image.new("RGB", (cols * cw, rows * ch))
    d = ImageDraw.Draw(im)
    for i, (fg, bg, g) in enumerate(cells):
        x, y = (i % cols) * cw, (i // cols) * ch
        d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=bg)
        if g != " ":
            d.text((x, y), g, fill=fg, font=font)
    im.save(out)

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
"#;
    std::fs::write(dir.join("render_gallery.py"), script).expect("write render_gallery.py");
}
