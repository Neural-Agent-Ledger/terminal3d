//! Small text-painting helpers shared by every screen: [`put_str`]/[`put_center`]
//! stamp a string into a [`re_core::Framebuffer`] at an explicit `fg`/`bg`;
//! [`fill_bg`] clears the whole framebuffer to one flat cell.
//!
//! Generalised from the game's original `put_str`/`put_center`/`clear_bg`
//! helpers to take `fg`/`bg` as explicit parameters instead of reading the
//! game's own `FG`/`BG` constants, so this crate has no dependency on any
//! game-specific palette. The game's `menu.rs` keeps thin `pub(crate)`
//! wrappers under the old names that delegate straight here, so every
//! existing call site across the game crate compiles unchanged.
//!
//! `fill_bg` takes BOTH colours, not just `bg`: `re_game`'s
//! `boot_console.rs::bleed_never_leaves_a_coloured_blank_cell` test asserts
//! the exact `fg` a freshly-cleared blank cell carries (proof that its
//! "bleed" corruption effect never touches an untouched cell), so the clear
//! must stay byte-identical to the pre-move `clear_bg` (`fg: FG, bg: BG`), not
//! collapse to a single colour. A caller that wants a truly flat field with
//! no distinct foreground can simply pass the same value twice.
//!
//! No ECS, no game state — every function here is a pure write into the
//! `Framebuffer` it is given.

use re_core::{Cell, Framebuffer};

/// Write `text` starting at `(col, row)`, one glyph per column, clipped to
/// `fb`'s own bounds, in `fg` over `bg`. Rows past the bottom of the
/// framebuffer are a no-op; columns past the right edge are dropped —
/// callers never need to pre-clip.
pub fn put_str(fb: &mut Framebuffer, col: usize, row: usize, text: &str, fg: u32, bg: u32) {
    if row >= fb.rows() {
        return;
    }
    for (i, ch) in text.chars().enumerate() {
        let c = col + i;
        if c >= fb.cols() {
            break;
        }
        fb.set(
            c,
            row,
            Cell {
                glyph: u32::from(ch),
                fg,
                bg,
                ..Default::default()
            },
        );
    }
}

/// Centre `text` horizontally on `row`, in `fg` over `bg` — the same
/// clipping guarantees as [`put_str`], which this delegates to after
/// computing the centred starting column.
pub fn put_center(fb: &mut Framebuffer, row: usize, text: &str, fg: u32, bg: u32) {
    let len = text.chars().count();
    let col = fb.cols().saturating_sub(len) / 2;
    put_str(fb, col, row, text, fg, bg);
}

/// Fill the whole framebuffer with blank (space) cells of `fg` over `bg`.
/// See the module doc for why this takes both colours rather than one.
pub fn fill_bg(fb: &mut Framebuffer, fg: u32, bg: u32) {
    let blank = Cell {
        glyph: u32::from(' '),
        fg,
        bg,
        ..Default::default()
    };
    for row in 0..fb.rows() {
        for col in 0..fb.cols() {
            fb.set(col, row, blank);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_str_clips_at_the_right_edge_without_panicking() {
        let mut fb = Framebuffer::new(5, 2);
        put_str(&mut fb, 3, 0, "HELLO", 0x00FF_FFFF, 0x0000_0000);
        assert_eq!(fb.get(3, 0).unwrap().glyph, u32::from('H'));
        assert_eq!(fb.get(4, 0).unwrap().glyph, u32::from('E'));
        // "LLO" fell off the right edge — must not have panicked, and must
        // not have wrapped onto another row.
        assert_eq!(fb.get(0, 1).unwrap().glyph, u32::from(' '));
    }

    #[test]
    fn put_str_below_the_last_row_is_a_no_op() {
        let mut fb = Framebuffer::new(5, 2);
        put_str(&mut fb, 0, 5, "X", 0x00FF_FFFF, 0x0000_0000);
        for row in 0..2 {
            for col in 0..5 {
                assert_eq!(fb.get(col, row).unwrap().glyph, u32::from(' '));
            }
        }
    }

    #[test]
    fn put_center_centres_odd_and_even_length_text() {
        let mut fb = Framebuffer::new(10, 1);
        put_center(&mut fb, 0, "AB", 0x00FF_FFFF, 0x0000_0000); // len 2 -> col 4
        assert_eq!(fb.get(4, 0).unwrap().glyph, u32::from('A'));
        assert_eq!(fb.get(5, 0).unwrap().glyph, u32::from('B'));
    }

    #[test]
    fn fill_bg_overwrites_every_cell_with_the_given_fg_and_bg() {
        let mut fb = Framebuffer::new(4, 3);
        put_str(&mut fb, 0, 0, "X", 0x00FF_0000, 0x0000_0000);
        fill_bg(&mut fb, 0x00AA_BBCC, 0x0012_3456);
        for row in 0..3 {
            for col in 0..4 {
                let cell = fb.get(col, row).unwrap();
                assert_eq!(cell.glyph, u32::from(' '));
                assert_eq!(cell.bg, 0x0012_3456);
                assert_eq!(cell.fg, 0x00AA_BBCC, "fg and bg must stay independent");
            }
        }
    }
}
