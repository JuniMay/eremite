//! The view 1 Overview: the work of the whole system.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::Line;

use super::draw::{
    build_section_rule, build_sparkline, colored, fit_line, format_percent, label, plain, strong,
};
use super::frame::Screen;
use super::samples::{CHART_COLUMNS, Sample, find_gpu_busy_since_start, find_ssd_read_since_start};
use super::theme::Rgb;

/// The columns of a figure's name in the System table.
const NAME_COLUMNS: usize = 24;
/// The columns of each figure in the System table.
const FIGURE_COLUMNS: usize = 16;
/// The history's sparkline takes two samples a column: half the columns of the
/// history charts.
const SAMPLES_PER_SPARKLINE_COLUMN: usize = 2;

/// A row of the System table: a figure now, recently, since the start, and
/// its history.
struct Figure {
    /// The figure's name.
    name: &'static str,
    /// The figure in a sample, if the sample has it.
    of: fn(&Sample) -> Option<f64>,
    /// The figure since the start.
    since_start: Option<f64>,
    /// Shows a value with its unit.
    show: fn(f64) -> String,
    /// The color of its sparkline.
    tone: Rgb,
    /// Whether its sparkline's scale starts at 0. A share's starts at its
    /// least value.
    from_zero: bool,
}

/// Draws 1 Overview in `area`.
pub fn draw_overview(buf: &mut Buffer, area: Rect, screen: &Screen) {
    let lines = build_system_lines(screen, area.width as usize);
    for (row, line) in lines.into_iter().take(area.height as usize).enumerate() {
        let line = fit_line(screen.theme, line, area.width as usize);
        buf.set_line(area.x, area.y + row as u16, &line, area.width);
    }
}

/// Builds the System section in `width` columns: the GPU's busy share and the
/// SSD's reads now, recently, since the start and in the history.
fn build_system_lines(screen: &Screen, width: usize) -> Vec<Line<'static>> {
    let theme = screen.theme;
    let samples = screen.samples;
    let windows = samples.windows;
    let mut lines = vec![build_section_rule(theme, width, "System")];
    let mut head = vec![plain(" ".repeat(1 + NAME_COLUMNS))];
    for name in ["now", "recent", "since start"] {
        head.push(label(theme, format!("{name:<FIGURE_COLUMNS$}")));
    }
    head.push(label(theme, "history"));
    lines.push(Line::from(head));
    let figures = [
        Figure {
            name: "GPU busy",
            of: |sample| sample.gpu_busy,
            since_start: find_gpu_busy_since_start(screen.statuses),
            show: |share| format_percent(share, 1.0),
            tone: theme.accent,
            from_zero: false,
        },
        Figure {
            name: "SSD read",
            of: |sample| sample.ssd_read,
            since_start: find_ssd_read_since_start(screen.statuses),
            show: |rate| format!("{rate:.1} GB/s"),
            tone: theme.sources[4],
            from_zero: true,
        },
    ];
    let sparkline_columns = width
        .saturating_sub(1 + NAME_COLUMNS + 3 * FIGURE_COLUMNS)
        .min(CHART_COLUMNS / SAMPLES_PER_SPARKLINE_COLUMN);
    // The history's samples, the newest at the right: the columns of samples
    // not taken yet stay blank.
    let newest = samples.samples.len().saturating_sub(CHART_COLUMNS);
    let blank = CHART_COLUMNS.saturating_sub(samples.samples.len());
    for figure in &figures {
        let mut line = vec![
            plain(" "),
            strong(format!("{:<NAME_COLUMNS$}", figure.name)),
        ];
        for value in [
            samples.find_mean(windows.now, figure.of),
            samples.find_mean(windows.recent, figure.of),
            figure.since_start,
        ] {
            line.push(match value {
                Some(value) => plain(format!("{:<FIGURE_COLUMNS$}", (figure.show)(value))),
                None => label(theme, format!("{:<FIGURE_COLUMNS$}", "—")),
            });
        }
        let mut values = vec![None; blank];
        values.extend(samples.samples.iter().skip(newest).map(figure.of));
        let sparkline = build_sparkline(&values, sparkline_columns, figure.from_zero);
        line.push(colored(theme, figure.tone, sparkline));
        lines.push(Line::from(line));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{Kind, ModelInfo};
    use crate::ui::samples::Samples;
    use crate::ui::status::ModelStatus;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, find_in_row, read_row};
    use crate::ui::theme::DARK;

    /// deepseek-v4.1-flash's status as the worker builds it, while request 5
    /// decodes, with `decode_seconds`, `gpu_seconds` and `read_bytes` so far.
    fn status(decode_seconds: f64, gpu_seconds: f64, read_bytes: u64) -> ModelStatus {
        let json = format!(
            r#"{{
            "loaded": true, "phase": "decode", "phase_seconds": 10.0,
            "requests": [{{"request": 5, "status": "decoding", "session": 5,
                           "prompt_tokens": 23, "reused_tokens": 2, "generated_tokens": 516}}],
            "state_disk": {{"dir": "/home/me/.cache/eremite/states", "gib": 0.0, "limit_gib": 32.0}},
            "model_memory": [{{"model": "deepseek-v4.1-flash", "weights": 7730941132,
                               "experts": 47352328192, "workspace": 64424509, "states": 354334801}}],
            "model_decode": {{"computed_rows": 708, "seconds": {decode_seconds},
                              "main_queue_gpu_seconds": {gpu_seconds},
                              "requested_expert_gb": 200.0, "read_wait_seconds": 8.6}},
            "totals": {{"prefill_seconds": 0.0}},
            "engine_counters": {{"layers": 43, "expert_read_bytes": {read_bytes},
                "mapped_expert_bytes": 45742809088,
                "memory": {{"footprint": 55942467584, "available": 11489037926,
                            "gpu_in_use": 56693760819, "gpu_limit": 115427033088,
                            "total": 137438953472, "free": 1181116006, "file": 7838315315,
                            "pressure": 1}}}}
            }}"#
        );
        serde_json::from_str(&json).expect("the worker's form")
    }

    /// Draws 1 Overview at `width` x `height` after two samples half a second
    /// apart.
    fn draw(width: u16, height: u16) -> Buffer {
        let models = [ModelInfo {
            name: "deepseek-v4.1-flash".into(),
            kind: Kind::Llm,
        }];
        let mut samples = Samples::new(WINDOWS);
        samples.add_sample(0.0, &[status(30.5, 20.4, 491_100_000_000)]);
        let statuses = [status(31.0, 20.8, 492_500_000_000)];
        samples.add_sample(0.5, &statuses);
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 57.0,
            models: &models,
            statuses: &statuses,
            samples: &samples,
        };
        draw_on_terminal(width, height, |buf, area| draw_overview(buf, area, &screen))
    }

    #[test]
    fn the_system_shows_the_gpus_busy_share_and_the_ssds_reads() {
        let buf = draw(160, 45);
        assert!(read_row(&buf, 0).starts_with("─ System ───"));
        assert_eq!(find_in_row(&buf, 1, "now"), Some(25));
        assert_eq!(find_in_row(&buf, 1, "history"), Some(73));
        // 0.4 GPU seconds in 0.5 s; since the start 20.8 in 31.0
        let gpu_busy = read_row(&buf, 2);
        assert!(
            gpu_busy.starts_with(" GPU busy                80.0%           80.0%           67.1%"),
            "{gpu_busy}"
        );
        // 1.4 GB in 0.5 s
        assert!(read_row(&buf, 3).contains("2.8 GB/s        2.8 GB/s"));
        // the two samples fill the newest column of the sparkline, in red:
        // the one rate is the top of its scale
        assert_eq!(buf[(73 + 59, 3)].symbol(), "█");
        assert_eq!(buf[(73 + 59, 3)].fg, DARK.color(DARK.sources[4]));
        assert_eq!(buf[(73 + 58, 3)].symbol(), " ");
    }

    #[test]
    fn the_system_fits_narrow_and_tiny_terminals() {
        let buf = draw(100, 30);
        assert!(read_row(&buf, 2).starts_with(" GPU busy                80.0%"));
        let buf = draw(30, 4);
        assert!(read_row(&buf, 0).starts_with("─ System ─"));
        // a terminal of one cell takes the view without a panic
        draw(1, 1);
    }
}
