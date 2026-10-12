//! Minimal text painting into a Framebuffer (shared by overlays + ts_editor).
use crate::{Cell, Framebuffer};

/// Paint `s` starting at (col,row), one cell per char, clipped to the right edge.
pub fn draw_str(fb: &mut Framebuffer, col: usize, row: usize, s: &str, fg: u32, bg: u32) {
    for (i, ch) in s.chars().enumerate() {
        let c = col + i;
        if c >= fb.cols() || row >= fb.rows() {
            break;
        }
        let cell = Cell {
            glyph: ch as u32,
            fg,
            bg,
            ..Cell::default()
        };
        fb.set(c, row, cell);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draws_glyphs_and_clips() {
        let mut fb = Framebuffer::new(4, 2);
        draw_str(&mut fb, 1, 0, "ABCDEF", 0x00FF00, 0x000000);
        assert_eq!(fb.get(1, 0).unwrap().glyph, 'A' as u32);
        assert_eq!(fb.get(3, 0).unwrap().glyph, 'C' as u32); // clipped at col 4
    }
}
