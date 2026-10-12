//! HTML photo + report.html + report.json writers — §6.
//!
//! All hand-rolled (no template engine, no serde). A frame "photo" is a truecolor
//! `<pre>` of `<span style="color:#fg;background:#bg">glyph</span>`. The report is
//! a JS-sortable table; the JSON is emitted by a tiny local serializer.

use std::fmt::Write as _;

use re_core::Framebuffer;

use crate::metrics::FrameMetrics;
use crate::static_map::StaticReport;
use crate::timing::Timing;

/// HTML-escape a single glyph (`<`, `>`, `&`, plus the attribute-sensitive `"`).
/// Non-printable / control codepoints render as a space so the photo never breaks.
pub fn escape_glyph(glyph: u32) -> String {
    match glyph {
        0x26 => "&amp;".to_string(),
        0x3C => "&lt;".to_string(),
        0x3E => "&gt;".to_string(),
        0x22 => "&quot;".to_string(),
        _ => match char::from_u32(glyph) {
            Some(' ') => " ".to_string(),
            Some(c) if c.is_control() => " ".to_string(),
            Some(c) => c.to_string(),
            None => " ".to_string(),
        },
    }
}

/// Escape text for HTML body content.
pub fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

#[inline]
fn hex6(packed: u32) -> String {
    format!("#{:06x}", packed & 0x00FF_FFFF)
}

/// Render one framebuffer to a standalone truecolor HTML photo document, with a
/// header line of the pose id + its metric row.
pub fn frame_photo(level: &str, pose_id: &str, fb: &Framebuffer, m: &FrameMetrics) -> String {
    let mut s = String::with_capacity(fb.cols() * fb.rows() * 48);
    s.push_str("<!doctype html><html><head><meta charset=\"utf-8\">");
    let _ = write!(
        s,
        "<title>{} · {}</title>",
        escape_text(level),
        escape_text(pose_id)
    );
    s.push_str("<style>body{background:#111;color:#ccc;font-family:monospace;margin:8px}");
    s.push_str("pre{line-height:1.0;letter-spacing:0;font-size:10px;white-space:pre}");
    s.push_str(".meta{color:#9cf;margin-bottom:6px;font-size:12px}</style></head><body>");

    let _ = write!(
        s,
        "<div class=\"meta\">{} · {} — sep={:.3} invis={:.1}% (near={:.1}% far={:.1}%) depthρ={:.2} spread={:.3} cvd={:.3}{}</div>",
        escape_text(level),
        escape_text(pose_id),
        m.separation,
        m.invisible_glyph_pct,
        m.invisible_pct_near,
        m.invisible_pct_far,
        m.depth_luma_spearman,
        m.luma_spread,
        m.cvd_min_boundary_contrast,
        degenerate_tag(m),
    );

    s.push_str("<pre>");
    let mut cur_fg = u32::MAX;
    let mut cur_bg = u32::MAX;
    let mut span_open = false;
    for row in 0..fb.rows() {
        for col in 0..fb.cols() {
            let c = fb.get(col, row).expect("in-bounds");
            if !span_open || c.fg != cur_fg || c.bg != cur_bg {
                if span_open {
                    s.push_str("</span>");
                }
                let _ = write!(
                    s,
                    "<span style=\"color:{};background:{}\">",
                    hex6(c.fg),
                    hex6(c.bg)
                );
                cur_fg = c.fg;
                cur_bg = c.bg;
                span_open = true;
            }
            s.push_str(&escape_glyph(c.glyph));
        }
        if span_open {
            s.push_str("</span>");
            span_open = false;
        }
        s.push('\n');
    }
    s.push_str("</pre></body></html>");
    s
}

fn degenerate_tag(m: &FrameMetrics) -> String {
    let mut tags = Vec::new();
    if m.is_blank {
        tags.push("BLANK");
    }
    if m.single_glyph {
        tags.push("SINGLE_GLYPH");
    }
    if m.single_color {
        tags.push("SINGLE_COLOR");
    }
    if tags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", tags.join(","))
    }
}

/// A row in the report table: which level + pose, its metrics, and the photo
/// link (or `None` if not photographed).
#[derive(Clone, Debug)]
pub struct ReportRow {
    /// The level's file stem (e.g. `"22-lit-cellar"`).
    pub level: String,
    /// The orbit pose identifier this row's metrics were computed from.
    pub pose_id: String,
    /// The computed readability/contrast metrics for this frame.
    pub metrics: FrameMetrics,
    /// Path to the saved HTML "photo" for this frame, relative to the report,
    /// or `None` if this frame was not selected for a photo.
    pub photo_rel: Option<String>,
}

/// Build the top-level sortable `report.html` from every analyzed frame plus the
/// per-level static reports and orbit shimmer summaries.
pub fn report_html(
    rows: &[ReportRow],
    statics: &[StaticReport],
    shimmer: &[(String, f32)],
    timing: &Timing,
) -> String {
    let mut s = String::with_capacity(rows.len() * 256 + 4096);
    s.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Render-QA report</title>",
    );
    s.push_str("<style>");
    s.push_str("body{background:#15171c;color:#cdd;font-family:system-ui,monospace;margin:14px}");
    s.push_str("h1,h2{color:#9cf}table{border-collapse:collapse;width:100%;font-size:12px}");
    s.push_str("th,td{border:1px solid #2c3038;padding:3px 6px;text-align:right}");
    s.push_str("th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}");
    s.push_str("th{cursor:pointer;background:#1e222a;position:sticky;top:0}");
    s.push_str("tr.deg{background:#3a1414}tr:hover{background:#222831}");
    s.push_str("a{color:#7cf}.flag{color:#f88;font-weight:bold}");
    s.push_str("section{margin:18px 0}.agg{color:#8d8}</style></head><body>");

    s.push_str("<h1>Render-QA report</h1>");
    let _ = write!(
        s,
        "<p>levels: {} · frames analyzed: {} · photos: {} · {:.0} frames/s · wall {}</p>",
        statics.len(),
        timing.frames_analyzed,
        timing.photos_written,
        timing.frames_per_sec(),
        crate::timing::fmt_ms(timing.wall_ns),
    );

    // ── frame table ──
    s.push_str("<section><h2>Frames</h2>");
    s.push_str("<table id=\"frames\"><thead><tr>");
    for (i, h) in [
        "level",
        "pose",
        "separation",
        "invisible%",
        "near%",
        "far%",
        "depthρ",
        "luma_spread",
        "cvd_min",
        "glyph_H",
        "color_H",
        "edge_v/h",
        "flags",
        "photo",
    ]
    .iter()
    .enumerate()
    {
        let _ = write!(s, "<th onclick=\"sortBy({i})\">{h}</th>");
    }
    s.push_str("</tr></thead><tbody>");

    // Default sort: degenerate first, then worst separation. Pre-sort here so the
    // table is meaningful even before JS runs.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        let da = is_degenerate(&rows[a].metrics);
        let db = is_degenerate(&rows[b].metrics);
        db.cmp(&da).then(
            rows[a]
                .metrics
                .separation
                .partial_cmp(&rows[b].metrics.separation)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    for &i in &order {
        let r = &rows[i];
        let m = &r.metrics;
        let deg = is_degenerate(m);
        let _ = write!(s, "<tr class=\"{}\">", if deg { "deg" } else { "" });
        let _ = write!(s, "<td>{}</td>", escape_text(&r.level));
        let _ = write!(s, "<td>{}</td>", escape_text(&r.pose_id));
        let _ = write!(s, "<td>{:.3}</td>", m.separation);
        let _ = write!(s, "<td>{:.1}</td>", m.invisible_glyph_pct);
        let _ = write!(s, "<td>{:.1}</td>", m.invisible_pct_near);
        let _ = write!(s, "<td>{:.1}</td>", m.invisible_pct_far);
        let _ = write!(s, "<td>{:.2}</td>", m.depth_luma_spearman);
        let _ = write!(s, "<td>{:.3}</td>", m.luma_spread);
        let _ = write!(s, "<td>{:.3}</td>", m.cvd_min_boundary_contrast);
        let _ = write!(s, "<td>{:.2}</td>", m.glyph_entropy);
        let _ = write!(s, "<td>{:.2}</td>", m.color_entropy);
        let _ = write!(s, "<td>{}</td>", fmt_ratio(m.edge_orientation_ratio));
        let _ = write!(s, "<td class=\"flag\">{}</td>", flags_text(m));
        match &r.photo_rel {
            Some(p) => {
                let _ = write!(s, "<td><a href=\"{}\">view</a></td>", escape_text(p));
            }
            None => s.push_str("<td>—</td>"),
        }
        s.push_str("</tr>");
    }
    s.push_str("</tbody></table></section>");

    // ── per-level aggregates ──
    s.push_str("<section><h2>Per-level aggregates</h2><table><thead><tr>");
    s.push_str("<th>level</th><th>frames</th><th>mean sep</th><th>min sep</th>");
    s.push_str("<th>mean invis%</th><th>mean near%</th><th>mean far%</th><th>min cvd</th><th>shimmer</th></tr></thead><tbody>");
    for st in statics {
        let lvl_rows: Vec<&ReportRow> = rows.iter().filter(|r| r.level == st.level).collect();
        let frames = lvl_rows.len();
        if frames == 0 {
            continue;
        }
        let mean_sep = lvl_rows.iter().map(|r| r.metrics.separation).sum::<f32>() / frames as f32;
        let min_sep = lvl_rows
            .iter()
            .map(|r| r.metrics.separation)
            .fold(f32::INFINITY, f32::min);
        let mean_invis = lvl_rows
            .iter()
            .map(|r| r.metrics.invisible_glyph_pct)
            .sum::<f32>()
            / frames as f32;
        let mean_near = lvl_rows
            .iter()
            .map(|r| r.metrics.invisible_pct_near)
            .sum::<f32>()
            / frames as f32;
        let mean_far = lvl_rows
            .iter()
            .map(|r| r.metrics.invisible_pct_far)
            .sum::<f32>()
            / frames as f32;
        let min_cvd = lvl_rows
            .iter()
            .map(|r| r.metrics.cvd_min_boundary_contrast)
            .fold(f32::INFINITY, f32::min);
        let shim = shimmer
            .iter()
            .find(|(l, _)| l == &st.level)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        let _ = write!(
            s,
            "<tr class=\"agg\"><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{:.3}</td><td>{:.4}</td></tr>",
            escape_text(&st.level), frames, mean_sep, min_sep, mean_invis, mean_near, mean_far, min_cvd, shim,
        );
    }
    s.push_str("</tbody></table></section>");

    // ── static reports ──
    s.push_str("<section><h2>Static map analysis</h2>");
    for st in statics {
        let _ = write!(s, "<h3>{}</h3><ul>", escape_text(&st.level));
        let _ = write!(
            s,
            "<li>reachable open cells: {}/{} ({})</li>",
            st.reachability.reachable_open,
            st.reachability.total_open,
            if st.reachability.all_interactive_reachable {
                "all interactive reachable".to_string()
            } else {
                format!(
                    "<span class=\"flag\">{} interactive UNREACHABLE</span>",
                    st.reachability.unreachable_interactive
                )
            }
        );
        let _ = write!(
            s,
            "<li>openness: mean {:.2}, median {:.1}, dead-ends {}</li>",
            st.openness.mean_open_neighbors,
            st.openness.median_open_neighbors,
            st.openness.dead_ends
        );
        let _ = write!(
            s,
            "<li>puzzle: {} nodes, {} pipes, depth {}, branching {:.2}, solvable {}, door-targets {}</li>",
            st.puzzle.node_count, st.puzzle.pipe_count, st.puzzle.max_depth,
            st.puzzle.branching_factor, st.puzzle.solvable, st.puzzle.door_targets,
        );
        if st.palette_collisions.is_empty() {
            s.push_str("<li>palette: no collisions</li>");
        } else {
            for c in &st.palette_collisions {
                let _ = write!(
                    s,
                    "<li class=\"flag\">palette collision: {} ↔ {} (ΔE {:.3})</li>",
                    escape_text(&c.a),
                    escape_text(&c.b),
                    c.delta_e
                );
            }
        }
        s.push_str("</ul>");
    }
    s.push_str("</section>");

    // ── sort script ──
    s.push_str(SORT_SCRIPT);
    s.push_str("</body></html>");
    s
}

fn fmt_ratio(r: f32) -> String {
    if r.is_infinite() {
        "∞".to_string()
    } else {
        format!("{r:.2}")
    }
}

fn is_degenerate(m: &FrameMetrics) -> bool {
    m.is_blank || m.single_glyph || m.single_color
}

fn flags_text(m: &FrameMetrics) -> String {
    let mut t = Vec::new();
    if m.is_blank {
        t.push("blank");
    }
    if m.single_glyph {
        t.push("1glyph");
    }
    if m.single_color {
        t.push("1color");
    }
    if m.invisible_glyph_pct > 25.0 {
        t.push("invis");
    }
    t.join(",")
}

/// Minimal column-sort script for the frame table (numeric where possible).
const SORT_SCRIPT: &str = r#"<script>
function sortBy(col){
  var tb=document.querySelector('#frames tbody');
  var rows=Array.prototype.slice.call(tb.querySelectorAll('tr'));
  var asc=tb.getAttribute('data-col')==col && tb.getAttribute('data-asc')!='1';
  rows.sort(function(a,b){
    var x=a.children[col].textContent.trim(), y=b.children[col].textContent.trim();
    var nx=parseFloat(x), ny=parseFloat(y);
    if(!isNaN(nx)&&!isNaN(ny)){return asc?nx-ny:ny-nx;}
    return asc? x.localeCompare(y): y.localeCompare(x);
  });
  rows.forEach(function(r){tb.appendChild(r);});
  tb.setAttribute('data-col',col); tb.setAttribute('data-asc',asc?'1':'0');
}
</script>"#;

// ───────────────────────────── JSON writer ──────────────────────────────────

/// A tiny JSON builder — just enough for the report. No external crate.
struct Json {
    s: String,
}

impl Json {
    fn new() -> Self {
        Json { s: String::new() }
    }
    fn raw(&mut self, v: &str) {
        self.s.push_str(v);
    }
    fn str_val(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }
}

fn jf(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else if v.is_infinite() && v > 0.0 {
        "1e999".to_string() // JSON has no Infinity; use an over-range literal
    } else if v.is_infinite() {
        "-1e999".to_string()
    } else {
        "null".to_string()
    }
}

fn metrics_json(m: &FrameMetrics) -> String {
    format!(
        "{{\"separation\":{},\"boundary_color_contrast\":{},\"boundary_shape_contrast\":{},\
\"interior_color_contrast\":{},\"interior_shape_contrast\":{},\"invisible_glyph_pct\":{},\
\"invisible_pct_near\":{},\"invisible_pct_far\":{},\"invis_near_depth\":{},\
\"depth_luma_spearman\":{},\"luma_p5\":{},\"luma_p95\":{},\"luma_spread\":{},\
\"distinct_glyphs\":{},\"distinct_colors\":{},\"hue_spread\":{},\"glyph_entropy\":{},\
\"color_entropy\":{},\"edge_orientation_ratio\":{},\"is_blank\":{},\"single_glyph\":{},\
\"single_color\":{},\"cvd_min_boundary_contrast\":{},\"unmapped_glyphs\":{}}}",
        jf(m.separation),
        jf(m.boundary_color_contrast),
        jf(m.boundary_shape_contrast),
        jf(m.interior_color_contrast),
        jf(m.interior_shape_contrast),
        jf(m.invisible_glyph_pct),
        jf(m.invisible_pct_near),
        jf(m.invisible_pct_far),
        jf(m.invis_near_depth),
        jf(m.depth_luma_spearman),
        jf(m.luma_p5),
        jf(m.luma_p95),
        jf(m.luma_spread),
        m.distinct_glyphs,
        m.distinct_colors,
        jf(m.hue_spread),
        jf(m.glyph_entropy),
        jf(m.color_entropy),
        jf(m.edge_orientation_ratio),
        m.is_blank,
        m.single_glyph,
        m.single_color,
        jf(m.cvd_min_boundary_contrast),
        m.unmapped_glyphs,
    )
}

/// Emit the machine-readable `report.json`: every frame's metrics, per-level
/// static reports + shimmer, and the timing block.
pub fn report_json(
    rows: &[ReportRow],
    statics: &[StaticReport],
    shimmer: &[(String, f32)],
    timing: &Timing,
) -> String {
    let mut j = Json::new();
    j.raw("{\"frames\":[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            j.raw(",");
        }
        j.raw("{\"level\":");
        j.raw(&Json::str_val(&r.level));
        j.raw(",\"pose\":");
        j.raw(&Json::str_val(&r.pose_id));
        j.raw(",\"photo\":");
        match &r.photo_rel {
            Some(p) => j.raw(&Json::str_val(p)),
            None => j.raw("null"),
        }
        j.raw(",\"metrics\":");
        j.raw(&metrics_json(&r.metrics));
        j.raw("}");
    }
    j.raw("],\"static\":[");
    for (i, st) in statics.iter().enumerate() {
        if i > 0 {
            j.raw(",");
        }
        let shim = shimmer
            .iter()
            .find(|(l, _)| l == &st.level)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        let mut collisions = String::from("[");
        for (k, c) in st.palette_collisions.iter().enumerate() {
            if k > 0 {
                collisions.push(',');
            }
            collisions.push_str(&format!(
                "{{\"a\":{},\"b\":{},\"delta_e\":{}}}",
                Json::str_val(&c.a),
                Json::str_val(&c.b),
                jf(c.delta_e)
            ));
        }
        collisions.push(']');
        j.raw(&format!(
            "{{\"level\":{},\"shimmer\":{},\"reachability\":{{\"reachable_open\":{},\"total_open\":{},\"all_interactive_reachable\":{},\"unreachable_interactive\":{}}},\
\"openness\":{{\"mean_open_neighbors\":{},\"median_open_neighbors\":{},\"dead_ends\":{}}},\
\"puzzle\":{{\"node_count\":{},\"pipe_count\":{},\"max_depth\":{},\"branching_factor\":{},\"solvable\":{},\"door_targets\":{}}},\
\"palette_collisions\":{}}}",
            Json::str_val(&st.level),
            jf(shim),
            st.reachability.reachable_open,
            st.reachability.total_open,
            st.reachability.all_interactive_reachable,
            st.reachability.unreachable_interactive,
            jf(st.openness.mean_open_neighbors),
            jf(st.openness.median_open_neighbors),
            st.openness.dead_ends,
            st.puzzle.node_count,
            st.puzzle.pipe_count,
            st.puzzle.max_depth,
            jf(st.puzzle.branching_factor),
            st.puzzle.solvable,
            st.puzzle.door_targets,
            collisions,
        ));
    }
    j.raw("],\"timing\":");
    j.raw(&format!(
        "{{\"wall_ns\":{},\"raycast_ns\":{},\"analyze_ns\":{},\"html_ns\":{},\"frames_analyzed\":{},\"photos_written\":{},\"frames_per_sec\":{},\"per_level\":[",
        timing.wall_ns,
        timing.stages.raycast_ns,
        timing.stages.analyze_ns,
        timing.stages.html_ns,
        timing.frames_analyzed,
        timing.photos_written,
        jf(timing.frames_per_sec() as f32),
    ));
    for (i, (lvl, ns)) in timing.per_level_ns.iter().enumerate() {
        if i > 0 {
            j.raw(",");
        }
        j.raw(&format!(
            "{{\"level\":{},\"wall_ns\":{}}}",
            Json::str_val(lvl),
            ns
        ));
    }
    j.raw("]}}");
    j.s
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{Cell, SurfaceKind};

    fn demo_metrics() -> FrameMetrics {
        FrameMetrics {
            boundary_color_contrast: 0.3,
            boundary_shape_contrast: 0.4,
            interior_color_contrast: 0.05,
            interior_shape_contrast: 0.02,
            separation: 0.25,
            invisible_glyph_pct: 1.5,
            invisible_pct_near: 0.5,
            invisible_pct_far: 3.0,
            invis_near_depth: 8.0,
            depth_luma_spearman: -0.7,
            luma_p5: 0.1,
            luma_p95: 0.8,
            luma_spread: 0.7,
            distinct_glyphs: 9,
            distinct_colors: 40,
            hue_spread: 0.3,
            glyph_entropy: 2.1,
            color_entropy: 4.0,
            edge_orientation_ratio: 2.5,
            is_blank: false,
            single_glyph: false,
            single_color: false,
            cvd_min_boundary_contrast: 0.2,
            unmapped_glyphs: 0,
        }
    }

    #[test]
    fn photo_escapes_special_glyphs() {
        let mut fb = Framebuffer::new(3, 1);
        fb.set(
            0,
            0,
            Cell {
                glyph: u32::from('<'),
                fg: 0x00FF0000,
                bg: 0,
                surface: SurfaceKind::WallSolid,
                ..Cell::default()
            },
        );
        fb.set(
            1,
            0,
            Cell {
                glyph: u32::from('>'),
                fg: 0x0000FF00,
                bg: 0,
                ..Cell::default()
            },
        );
        fb.set(
            2,
            0,
            Cell {
                glyph: u32::from('&'),
                fg: 0x000000FF,
                bg: 0,
                ..Cell::default()
            },
        );
        let html = frame_photo("lvl", "spawn", &fb, &demo_metrics());
        assert!(html.contains("&lt;"), "must escape <");
        assert!(html.contains("&gt;"), "must escape >");
        assert!(html.contains("&amp;"), "must escape &");
        // No raw unescaped angle bracket from the glyph leaks into a text node.
        assert!(html.contains("<pre>") && html.contains("</pre>"));
        assert!(html.trim_end().ends_with("</html>"));
    }

    #[test]
    fn escape_glyph_basic() {
        assert_eq!(escape_glyph(u32::from('<')), "&lt;");
        assert_eq!(escape_glyph(u32::from('A')), "A");
        assert_eq!(escape_glyph(0x09), " "); // tab → space
    }

    #[test]
    fn report_json_contains_timing_and_metrics() {
        let rows = vec![ReportRow {
            level: "lvl".into(),
            pose_id: "spawn".into(),
            metrics: demo_metrics(),
            photo_rel: Some("lvl/spawn.html".into()),
        }];
        let statics = vec![];
        let timing = Timing {
            wall_ns: 1_000_000,
            frames_analyzed: 1,
            photos_written: 1,
            ..Timing::default()
        };
        let json = report_json(&rows, &statics, &[], &timing);
        assert!(
            json.contains("\"timing\""),
            "json must contain a timing block"
        );
        assert!(
            json.contains("\"separation\":0.250000"),
            "metrics must round-trip: {json}"
        );
        assert!(json.contains("\"frames_analyzed\":1"));
        // Valid-ish structure: balanced braces.
        let opens = json.matches('{').count();
        let closes = json.matches('}').count();
        assert_eq!(opens, closes, "JSON braces must balance");
    }
}
