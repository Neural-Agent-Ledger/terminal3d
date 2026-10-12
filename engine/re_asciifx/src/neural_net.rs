//! NEURAL NET — a small feed-forward node/edge diagram with a travelling
//! signal pulse on every edge (own speed per edge) and an independent
//! activation blink per node (own frequency/phase per node). A deliberate
//! nod to the game's own canon — TERMINAL STATE's puzzle IS a binary
//! neural-network forward-pass — rendered here as pure decoration rather
//! than anything that reads `re_puzzle` state (this module stays engine/
//! game-logic free, see the crate-level module doc).

use crate::hash::{hash_unit, range};
use crate::{secs, set_clipped, AsciiEffect, Rect};
use re_core::{Cell, Framebuffer};
use std::time::Duration;

const LAYER_COUNT_RANGE: (usize, usize) = (3, 5);
const NODES_PER_LAYER_RANGE: (usize, usize) = (2, 6);
const JITTER_ROWS_RANGE: (f32, f32) = (-1.5, 1.5);
const EDGE_SPEED_RANGE: (f32, f32) = (4.0, 16.0); // cells / second
const EDGE_PHASE_RANGE: (f32, f32) = (0.0, 1.0);
const NODE_FREQ_RANGE: (f32, f32) = (0.3, 2.5);
const NODE_PHASE_RANGE: (f32, f32) = (0.0, 1.0);
const NODE_DUTY_RANGE: (f32, f32) = (0.15, 0.45);
/// How many trailing cells of an edge's path read as "lit" behind the pulse
/// head, giving it a short comet tail instead of a single bright dot.
const PULSE_TAIL: i64 = 4;

const EDGE_DIM: u32 = 0x0022_4030;
const EDGE_LIT: u32 = 0x0060_FFA0;
const EDGE_GLOW_BG: u32 = 0x0007_140C;
const NODE_IDLE: u32 = 0x0030_4838;
const NODE_FIRING: u32 = 0x00A0_FFC0;
const NODE_HALO: u32 = 0x0020_5040;

/// The directional line glyph for a unit step `(dc, dr)` between consecutive
/// Bresenham path cells — a clear line (horizontal/vertical/diagonal rule)
/// instead of a repeated middle-dot, which read as "faint and dotted" in an
/// earlier version.
fn step_glyph(dc: i64, dr: i64) -> char {
    match (dc.signum(), dr.signum()) {
        (0, _) => '\u{2502}',      // │ vertical
        (_, 0) => '\u{2500}',      // ─ horizontal
        (1, 1) | (-1, -1) => '\\', // falling diagonal
        _ => '/',                  // rising diagonal
    }
}

const S_NODE_COUNT: u64 = 1;
const S_JITTER: u64 = 2;
const S_EDGE_KEEP: u64 = 3;
const S_EDGE_SPEED: u64 = 4;
const S_EDGE_PHASE: u64 = 5;
const S_NODE_FREQ: u64 = 6;
const S_NODE_PHASE: u64 = 7;
const S_NODE_DUTY: u64 = 8;

/// Keep-probability for a candidate edge between consecutive layers — not
/// every node connects to every node in the next layer, so the diagram
/// reads as a network rather than a solid grid.
const EDGE_KEEP_PROB: f32 = 0.55;

struct Node {
    col: f32,
    row: f32,
    freq: f32,
    phase: f32,
    duty: f32,
}

struct Edge {
    /// `(col, row, glyph)` — `glyph` is the directional line rule for the
    /// step INTO this cell from the previous one (the first cell reuses the
    /// second cell's glyph, since it has no "previous" step of its own).
    path: Vec<(i64, i64, char)>,
    speed: f32,
    phase: f32,
}

/// A small procedural feed-forward network diagram.
pub struct NeuralNet {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

fn bresenham(c0: i64, r0: i64, c1: i64, r1: i64) -> Vec<(i64, i64)> {
    let mut path = Vec::new();
    let dc = (c1 - c0).abs();
    let dr = (r1 - r0).abs();
    let sc = if c0 < c1 { 1 } else { -1 };
    let sr = if r0 < r1 { 1 } else { -1 };
    let mut err = dc - dr;
    let (mut c, mut r) = (c0, r0);
    loop {
        path.push((c, r));
        if c == c1 && r == r1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dr {
            err -= dr;
            c += sc;
        }
        if e2 < dc {
            err += dc;
            r += sr;
        }
    }
    path
}

/// Attach a directional line glyph to every cell of a raw Bresenham path
/// (see [`step_glyph`]); the first cell copies the second cell's glyph since
/// it has no incoming step of its own.
fn glyph_path(raw: &[(i64, i64)]) -> Vec<(i64, i64, char)> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(raw.len());
    for w in raw.windows(2) {
        let (c0, r0) = w[0];
        let (c1, r1) = w[1];
        out.push((c0, r0, step_glyph(c1 - c0, r1 - r0)));
    }
    let last = raw[raw.len() - 1];
    let last_glyph = out.last().map_or('\u{2500}', |&(_, _, g)| g);
    out.push((last.0, last.1, last_glyph));
    out
}

impl NeuralNet {
    /// Build a deterministic layered network from `seed`, sized and laid out
    /// over `rect`: a random layer count, a random node count per layer, and
    /// edges pruned probabilistically between adjacent layers only.
    #[must_use]
    pub fn random(seed: u64, rect: Rect) -> Self {
        let layer_count = LAYER_COUNT_RANGE.0
            + (hash_unit(seed, 0, 0) * (LAYER_COUNT_RANGE.1 - LAYER_COUNT_RANGE.0 + 1) as f64)
                as usize;
        let layer_count = layer_count
            .min(LAYER_COUNT_RANGE.1)
            .max(LAYER_COUNT_RANGE.0);

        let mut layer_sizes = Vec::with_capacity(layer_count);
        for l in 0..layer_count {
            let n = NODES_PER_LAYER_RANGE.0
                + (hash_unit(seed, l as u64, S_NODE_COUNT)
                    * (NODES_PER_LAYER_RANGE.1 - NODES_PER_LAYER_RANGE.0 + 1) as f64)
                    as usize;
            layer_sizes.push(n.min(NODES_PER_LAYER_RANGE.1).max(NODES_PER_LAYER_RANGE.0));
        }

        let mut nodes = Vec::new();
        let mut layer_start = Vec::with_capacity(layer_count);
        for (l, &n) in layer_sizes.iter().enumerate() {
            layer_start.push(nodes.len());
            let col = rect.col0 as f32 + (l as f32 + 0.5) / layer_count as f32 * rect.cols as f32;
            for k in 0..n {
                let id = (layer_start[l] + k) as u64;
                let row_frac = (k as f32 + 0.5) / n as f32;
                let jitter = range(seed, id, S_JITTER, JITTER_ROWS_RANGE.0, JITTER_ROWS_RANGE.1);
                let row = rect.row0 as f32 + row_frac * rect.rows as f32 + jitter;
                nodes.push(Node {
                    col,
                    row,
                    freq: range(seed, id, S_NODE_FREQ, NODE_FREQ_RANGE.0, NODE_FREQ_RANGE.1),
                    phase: range(
                        seed,
                        id,
                        S_NODE_PHASE,
                        NODE_PHASE_RANGE.0,
                        NODE_PHASE_RANGE.1,
                    ),
                    duty: range(seed, id, S_NODE_DUTY, NODE_DUTY_RANGE.0, NODE_DUTY_RANGE.1),
                });
            }
        }

        let mut edges = Vec::new();
        for l in 0..layer_count.saturating_sub(1) {
            let a0 = layer_start[l];
            let a1 = layer_start.get(l + 1).copied().unwrap_or(nodes.len());
            let b1 = layer_start.get(l + 2).copied().unwrap_or(nodes.len());
            for i in a0..a1 {
                for j in a1..b1 {
                    let eid = (i as u64) << 24 | j as u64;
                    if hash_unit(seed, eid, S_EDGE_KEEP) as f32 > EDGE_KEEP_PROB {
                        continue;
                    }
                    let (c0, r0) = (nodes[i].col.round() as i64, nodes[i].row.round() as i64);
                    let (c1, r1) = (nodes[j].col.round() as i64, nodes[j].row.round() as i64);
                    let raw = bresenham(c0, r0, c1, r1);
                    let path = glyph_path(&raw);
                    edges.push(Edge {
                        path,
                        speed: range(
                            seed,
                            eid,
                            S_EDGE_SPEED,
                            EDGE_SPEED_RANGE.0,
                            EDGE_SPEED_RANGE.1,
                        ),
                        phase: range(
                            seed,
                            eid,
                            S_EDGE_PHASE,
                            EDGE_PHASE_RANGE.0,
                            EDGE_PHASE_RANGE.1,
                        ),
                    });
                }
            }
        }

        NeuralNet { nodes, edges }
    }

    /// Total node count across every layer.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges kept after the probabilistic pruning pass.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl AsciiEffect for NeuralNet {
    fn name(&self) -> &'static str {
        "neural_net"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);

        // Firing nodes' glow halo, painted FIRST so the edges/node discs
        // layer cleanly on top of it — a soft "glowing disc" reads as a
        // bright center fading into its background, not a bare bullet glyph.
        for node in &self.nodes {
            let phase = (secs_t * node.freq + node.phase).rem_euclid(1.0);
            if phase >= node.duty {
                continue;
            }
            let (nc, nr) = (node.col.round() as i64, node.row.round() as i64);
            for (dc, dr) in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),
            ] {
                set_clipped(
                    fb,
                    rect,
                    nc + dc,
                    nr + dr,
                    Cell {
                        glyph: u32::from(' '),
                        fg: 0x0000_0000,
                        bg: NODE_HALO,
                        ..Default::default()
                    },
                );
            }
        }

        for edge in &self.edges {
            let len = edge.path.len().max(1) as f32;
            let pulse_pos = ((secs_t * edge.speed + edge.phase * len).rem_euclid(len)) as i64;
            for (idx, &(col, row, glyph)) in edge.path.iter().enumerate() {
                let d = (idx as i64 - pulse_pos).rem_euclid(edge.path.len() as i64);
                let lit = d < PULSE_TAIL;
                let fade = if lit {
                    1.0 - d as f32 / PULSE_TAIL as f32
                } else {
                    0.0
                };
                let color = if lit {
                    lerp_color(EDGE_DIM, EDGE_LIT, fade)
                } else {
                    EDGE_DIM
                };
                let bg = if lit {
                    scale_color(EDGE_GLOW_BG, 1.0 + fade * 2.5)
                } else {
                    EDGE_GLOW_BG
                };
                set_clipped(
                    fb,
                    rect,
                    col,
                    row,
                    Cell {
                        glyph: u32::from(glyph),
                        fg: color,
                        bg,
                        ..Default::default()
                    },
                );
            }
        }
        for node in &self.nodes {
            let phase = (secs_t * node.freq + node.phase).rem_euclid(1.0);
            let firing = phase < node.duty;
            let (glyph, color) = if firing {
                ('\u{25CF}', NODE_FIRING)
            } else {
                ('\u{25CB}', NODE_IDLE)
            };
            let bg = if firing { NODE_HALO } else { 0x0000_0000 };
            set_clipped(
                fb,
                rect,
                node.col.round() as i64,
                node.row.round() as i64,
                Cell {
                    glyph: u32::from(glyph),
                    fg: color,
                    bg,
                    ..Default::default()
                },
            );
        }
    }
}

fn scale_color(c: u32, k: f32) -> u32 {
    let l = re_core::oklab::unpack_linear(c);
    re_core::oklab::pack_linear(re_core::oklab::LinRgb {
        r: (l.r * k).clamp(0.0, 4.0),
        g: (l.g * k).clamp(0.0, 4.0),
        b: (l.b * k).clamp(0.0, 4.0),
    })
}

fn lerp_color(a: u32, b: u32, k: f32) -> u32 {
    let la = re_core::oklab::unpack_linear(a);
    let lb = re_core::oklab::unpack_linear(b);
    re_core::oklab::pack_linear(re_core::oklab::lerp_linear(la, lb, k.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 90,
        rows: 30,
    };

    #[test]
    fn determinism_same_seed() {
        let a = NeuralNet::random(21, RECT);
        let b = NeuralNet::random(21, RECT);
        assert_eq!(a.node_count(), b.node_count());
        assert_eq!(a.edge_count(), b.edge_count());
    }

    #[test]
    fn different_seeds_diverge_in_topology_and_frequency() {
        let a = NeuralNet::random(1, RECT);
        let b = NeuralNet::random(2, RECT);
        assert!(
            a.node_count() != b.node_count() || a.edge_count() != b.edge_count(),
            "two seeds produced an identical network shape"
        );
        assert!(a
            .nodes
            .iter()
            .zip(b.nodes.iter())
            .any(|(x, y)| x.freq != y.freq));
    }

    #[test]
    fn network_is_non_trivial() {
        let net = NeuralNet::random(5, RECT);
        assert!(net.node_count() >= LAYER_COUNT_RANGE.0 * NODES_PER_LAYER_RANGE.0);
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        for g in [
            '\u{2500}', '\u{2502}', '/', '\\', '\u{25CF}', '\u{25CB}', ' ',
        ] {
            assert!(
                is_reliable(g),
                "neural_net glyph {g:?} is not in the reliable set"
            );
        }
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(100, 34);
        let rect = Rect::new(4, 2, 90, 30);
        let net = NeuralNet::random(9, rect);
        for ms in [0u64, 300, 3000, 9000] {
            net.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..34 {
            for col in 0..100 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }
}
