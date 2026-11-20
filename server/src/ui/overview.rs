//! The view 1 Overview: the work and the memory of the whole system, the
//! requests that run or wait, and the log.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};

use super::draw::{
    build_gauge, build_section_rule, build_sparkline, colored, fit_line, format_count,
    format_percent, label, measure_width, pad_spans, plain, strong,
};
use super::frame::{Choices, Screen};
use super::log::{LogLine, draw_log};
use super::samples::{CHART_COLUMNS, Sample, find_gpu_busy_since_start, find_ssd_read_since_start};
use super::status::{Activity, ModelStatus, Request};
use super::theme::{Rgb, Theme};

/// The bytes of one GiB.
const GIB: f64 = (1u64 << 30) as f64;
/// The columns of a request's number in the Requests list.
const REQUEST_COLUMNS: usize = 9;
/// The columns after the longest model name in the Requests list.
const REQUEST_GAP: usize = 3;
/// The columns of a request's session in the Requests list.
const SESSION_COLUMNS: usize = 9;
/// The columns of what a request does in the Requests list.
const STATUS_COLUMNS: usize = 28;
/// The columns of a request's speed in the Requests list.
const SPEED_COLUMNS: usize = 14;
/// The columns of a figure's name in the System table.
const NAME_COLUMNS: usize = 24;
/// The columns of each figure in the System table.
const FIGURE_COLUMNS: usize = 16;
/// The history's sparkline takes two samples a column: half the columns of the
/// history charts.
const SAMPLES_PER_SPARKLINE_COLUMN: usize = 2;

/// The cells of a memory row's gauge on a wide terminal, and on a narrow one.
const GAUGE_CELLS: (usize, usize) = (24, 10);
/// A terminal at least this wide takes the wide gauges. A memory row with a
/// wide gauge then has room for its figures.
const WIDE_COLUMNS: usize = 110;
/// The columns of a memory row's main figure.
const MAIN_FIGURE_COLUMNS: usize = 18;
/// The memory pressure that warns, as the engine reports it.
const PRESSURE_WARNS: u64 = 2;
/// The memory pressure that is critical, as the engine reports it.
const PRESSURE_CRITICAL: u64 = 4;

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

/// Draws 1 Overview in `area`: the System section, the requests that run or
/// wait, and the log of `log_lines` in the rows left. The log's filter and
/// scroll are in `choices`.
pub fn draw_overview(
    buf: &mut Buffer,
    area: Rect,
    screen: &Screen,
    log_lines: &[LogLine],
    choices: &mut Choices,
) {
    let width = area.width as usize;
    let mut lines = build_system_lines(screen, width);
    // The requests take at most half of the rows that the System section
    // leaves. The log keeps the rest.
    let left = (area.height as usize).saturating_sub(lines.len());
    lines.extend(build_request_lines(screen, width, left / 2));
    let shown = lines.len().min(area.height as usize);
    for (row, line) in lines.into_iter().take(shown).enumerate() {
        let line = fit_line(screen.theme, line, width);
        buf.set_line(area.x, area.y + row as u16, &line, area.width);
    }
    let log = Rect {
        y: area.y + shown as u16,
        height: area.height - shown as u16,
        ..area
    };
    draw_log(buf, log, screen, log_lines, choices);
}

/// Builds the Requests section in `width` columns and at most `rows` rows:
/// one row for each request that runs or waits, for all models. Each row
/// shows the request's number, its model, its session, what it does, its
/// speed and its tokens. A request that decodes shows its generated tokens.
/// One that prefills or waits shows its prompt's tokens.
fn build_request_lines(screen: &Screen, width: usize, rows: usize) -> Vec<Line<'static>> {
    let theme = screen.theme;
    let mut lines = vec![build_section_rule(theme, width, "Requests")];
    let requests: Vec<(usize, &Request)> = screen
        .statuses
        .iter()
        .enumerate()
        .flat_map(|(i, status)| status.requests.iter().map(move |request| (i, request)))
        .collect();
    if requests.is_empty() {
        lines.push(Line::from(vec![plain(" "), label(theme, "no requests")]));
        return lines;
    }
    let model_columns = screen
        .models
        .iter()
        .map(|model| model.name.chars().count())
        .chain(["model".len()])
        .max()
        .unwrap_or(0)
        + REQUEST_GAP;
    let head = format!(
        " {:<REQUEST_COLUMNS$}{:<model_columns$}{:<SESSION_COLUMNS$}{:<STATUS_COLUMNS$}{:<SPEED_COLUMNS$}amount",
        "request", "model", "session", "status", "speed"
    );
    lines.push(Line::from(label(theme, head)));
    // The header and the rule take two rows. A last row names the requests
    // that do not fit.
    let room = rows.saturating_sub(2);
    let fit = if requests.len() > room {
        room.saturating_sub(1)
    } else {
        requests.len()
    };
    let latest = screen.samples.samples.back();
    for &(i, request) in &requests[..fit] {
        let status = &screen.statuses[i];
        let name = screen.models.get(i).map_or("", |model| model.name.as_str());
        let session = request
            .session
            .map_or(String::new(), |number| number.to_string());
        let mut spans = vec![
            plain(" "),
            strong(format!("{:<REQUEST_COLUMNS$}", format!("#{}", request.id))),
            plain(format!("{name:<model_columns$}")),
            label(theme, format!("{session:<SESSION_COLUMNS$}")),
        ];
        let activity = describe_activity(theme, status, request);
        let activity_width = measure_width(&activity);
        spans.extend(activity);
        spans.push(plain(
            " ".repeat(STATUS_COLUMNS.saturating_sub(activity_width)),
        ));
        let (speed, tokens) = match request.status {
            Activity::Decoding => (
                screen
                    .samples
                    .find_request_speed(request.id)
                    .map(|speed| format!("{speed:.1} tok/s")),
                request.generated_tokens,
            ),
            Activity::Prefill => (
                latest
                    .and_then(|sample| sample.models.get(i))
                    .and_then(|model| model.prefill)
                    .map(|speed| format!("{} tok/s", format_count(speed))),
                request.prompt_tokens,
            ),
            Activity::Generating | Activity::Waiting => (None, request.prompt_tokens),
        };
        spans.push(plain(format!(
            "{:<SPEED_COLUMNS$}",
            speed.unwrap_or_default()
        )));
        let amount = if request.images_requested > 0 {
            let unit = if request.images_requested == 1 {
                "image"
            } else {
                "images"
            };
            format!(
                "{} of {} {unit}",
                request.generated_images, request.images_requested
            )
        } else {
            format!("{} tokens", format_count(tokens as f64))
        };
        spans.push(plain(amount));
        lines.push(Line::from(spans));
    }
    if fit < requests.len() {
        let more = format!(" … {} more", requests.len() - fit);
        lines.push(Line::from(label(theme, more)));
    }
    lines
}

/// Describes what `request`, a request of the model of `status`, does: it
/// waits in amber, decodes in the accent, prefills in the prefill color with
/// the layer reached, or generates images in green with the denoising step.
fn describe_activity(theme: &Theme, status: &ModelStatus, request: &Request) -> Vec<Span<'static>> {
    let counters = &status.engine_counters;
    match request.status {
        Activity::Waiting => vec![colored(theme, theme.amber, "waiting")],
        Activity::Decoding => vec![colored(theme, theme.accent, "decoding")],
        Activity::Prefill if counters.prefill_layer > 0 && counters.layers > 0 => vec![
            colored(theme, theme.prefill, "prefill"),
            label(theme, ", layer "),
            plain(counters.prefill_layer.to_string()),
            label(theme, " of "),
            plain(counters.layers.to_string()),
        ],
        Activity::Prefill => vec![colored(theme, theme.prefill, "prefill")],
        Activity::Generating if counters.image_step > 0 && counters.image_steps_total > 0 => vec![
            colored(theme, theme.green, "generating"),
            label(theme, ", step "),
            plain(counters.image_step.to_string()),
            label(theme, " of "),
            plain(counters.image_steps_total.to_string()),
        ],
        Activity::Generating => vec![colored(theme, theme.green, "generating")],
    }
}

/// Builds the System section in `width` columns: the GPU's busy share and the
/// SSD's reads now, recently, since the start and in the history; then the
/// memory of the GPU, of the server, of the saved states and of the Mac.
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
    let gauge_cells = if width >= WIDE_COLUMNS {
        GAUGE_CELLS.0
    } else {
        GAUGE_CELLS.1
    };
    lines.extend([
        build_gpu_row(screen, gauge_cells),
        build_ram_row(screen, gauge_cells),
        build_ssd_row(screen, gauge_cells),
        build_mac_row(screen, gauge_cells),
    ]);
    lines
}

/// Returns `bytes` in GiB with one decimal.
fn format_gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / GIB)
}

/// Returns `part` as a share of `whole`, or 0 for an empty whole.
fn share(part: u64, whole: u64) -> f64 {
    if whole > 0 {
        part as f64 / whole as f64
    } else {
        0.0
    }
}

/// Returns the start of a memory row: its name and its gauge of `parts`.
fn start_memory_row(
    theme: &Theme,
    name: &str,
    gauge_cells: usize,
    parts: &[(f64, Style)],
) -> Vec<Span<'static>> {
    let mut spans = vec![plain(" "), strong(name.to_string()), plain("  ")];
    spans.extend(build_gauge(theme, gauge_cells, parts));
    spans.push(plain("  "));
    spans
}

/// Returns a part of a memory row's legend: a square in `rgb`, the part's
/// name and its GiB.
fn build_legend_part(theme: &Theme, rgb: Rgb, name: &str, bytes: u64) -> Vec<Span<'static>> {
    vec![
        colored(theme, rgb, "■"),
        plain(" "),
        label(theme, name.to_string()),
        plain(format!(" {}   ", format_gib(bytes))),
    ]
}

/// Builds the GPU's row: the GPU's memory in use by all programs against its
/// limit. Its parts are all models' experts mapped for the GPU, all models'
/// weights, and the other memory in use.
fn build_gpu_row(screen: &Screen, gauge_cells: usize) -> Line<'static> {
    let theme = screen.theme;
    let Some(first) = screen.statuses.first() else {
        return Line::default();
    };
    let memory = &first.engine_counters.memory;
    let experts: u64 = screen
        .statuses
        .iter()
        .map(|status| status.engine_counters.mapped_expert_bytes)
        .sum();
    let weights: u64 = first.model_memory.iter().map(|model| model.weights).sum();
    let other = memory.gpu_in_use.saturating_sub(experts + weights);
    let limit = memory.gpu_limit.unwrap_or(0);
    let style = |rgb: Rgb| Style::new().fg(theme.color(rgb));
    let mut spans = start_memory_row(
        theme,
        "GPU",
        gauge_cells,
        &[
            (share(experts, limit), style(theme.sources[0])),
            (share(weights, limit), style(theme.sources[1])),
            (share(other, limit), Style::new()),
        ],
    );
    let mut figure = vec![strong(format_gib(memory.gpu_in_use))];
    if limit > 0 {
        figure.push(plain(format!(" / {} GiB", format_gib(limit))));
    } else {
        figure.push(plain(" GiB"));
    }
    spans.extend(pad_spans(figure, MAIN_FIGURE_COLUMNS));
    spans.extend(build_legend_part(
        theme,
        theme.sources[0],
        "experts",
        experts,
    ));
    spans.extend(build_legend_part(
        theme,
        theme.sources[1],
        "weights",
        weights,
    ));
    spans.extend([
        label(theme, "other"),
        plain(format!(" {}", format_gib(other))),
    ]);
    Line::from(spans)
}

/// Builds the RAM's row: the server's memory against the Mac's. Its parts are
/// all models' experts and the rest.
fn build_ram_row(screen: &Screen, gauge_cells: usize) -> Line<'static> {
    let theme = screen.theme;
    let Some(first) = screen.statuses.first() else {
        return Line::default();
    };
    let memory = &first.engine_counters.memory;
    let experts: u64 = first.model_memory.iter().map(|model| model.experts).sum();
    let rest = memory.footprint.saturating_sub(experts);
    let style = |rgb: Rgb| Style::new().fg(theme.color(rgb));
    let mut spans = start_memory_row(
        theme,
        "RAM",
        gauge_cells,
        &[
            (share(experts, memory.total), style(theme.sources[0])),
            (share(rest, memory.total), style(theme.sources[1])),
        ],
    );
    let figure = vec![
        label(theme, "eremite"),
        plain(" "),
        strong(format_gib(memory.footprint)),
        plain(" GiB"),
    ];
    spans.extend(pad_spans(figure, MAIN_FIGURE_COLUMNS));
    spans.extend(build_legend_part(
        theme,
        theme.sources[0],
        "experts",
        experts,
    ));
    spans.extend(build_legend_part(theme, theme.sources[1], "rest", rest));
    Line::from(spans)
}

/// Builds the SSD's row: the saved states' GiB against their limit, and their
/// folder, with the home folder as ~.
fn build_ssd_row(screen: &Screen, gauge_cells: usize) -> Line<'static> {
    let theme = screen.theme;
    let disk = screen
        .statuses
        .first()
        .and_then(|status| status.state_disk.as_ref());
    let Some(disk) = disk else {
        let mut spans = start_memory_row(theme, "SSD", gauge_cells, &[]);
        spans.push(label(theme, "no saved states"));
        return Line::from(spans);
    };
    let used = if disk.limit_gib > 0.0 {
        disk.gib / disk.limit_gib
    } else {
        0.0
    };
    let mut spans = start_memory_row(theme, "SSD", gauge_cells, &[(used, Style::new())]);
    let figure = vec![
        strong(format!("{:.2}", disk.gib)),
        plain(format!(" / {:.1} GiB", disk.limit_gib)),
    ];
    spans.extend(pad_spans(figure, MAIN_FIGURE_COLUMNS));
    spans.extend([
        label(theme, "saved states in "),
        plain(shorten_home(&disk.dir, screen.home)),
    ]);
    Line::from(spans)
}

/// Builds the Mac's row: a gauge of the memory in use, the memory available
/// and the memory pressure.
fn build_mac_row(screen: &Screen, gauge_cells: usize) -> Line<'static> {
    let theme = screen.theme;
    let Some(first) = screen.statuses.first() else {
        return Line::default();
    };
    let memory = &first.engine_counters.memory;
    let in_use = memory.total.saturating_sub(memory.available);
    let mut spans = start_memory_row(
        theme,
        "Mac",
        gauge_cells,
        &[(share(in_use, memory.total), Style::new())],
    );
    let pressure = if memory.pressure >= PRESSURE_CRITICAL {
        colored(theme, theme.red, "critical")
    } else if memory.pressure >= PRESSURE_WARNS {
        colored(theme, theme.amber, "warning")
    } else {
        colored(theme, theme.green, "normal")
    };
    spans.extend([
        label(theme, "available "),
        strong(format_gib(memory.available)),
        plain(" GiB   "),
        label(theme, "pressure "),
        pressure,
    ]);
    Line::from(spans)
}

/// Returns `path` with the home folder `home` at its start shown as ~.
fn shorten_home(path: &str, home: &str) -> String {
    match path.strip_prefix(home) {
        Some(rest) if !home.is_empty() && (rest.is_empty() || rest.starts_with('/')) => {
            format!("~{rest}")
        }
        _ => path.to_string(),
    }
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
            home: "/home/me",
            session_text: None,
        };
        draw_on_terminal(width, height, |buf, area| {
            draw_overview(buf, area, &screen, &[], &mut Choices::default());
        })
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
    fn the_system_shows_the_memory_of_all_models() {
        let buf = draw(160, 45);
        let gpu = read_row(&buf, 4);
        assert!(
            gpu.contains("  52.8 / 107.5 GiB  ■ experts 42.6   ■ weights 7.2   other 3.0"),
            "{gpu}"
        );
        // the gauge: experts in blue, weights in light blue, the rest empty
        assert_eq!(buf[(6, 4)].fg, DARK.color(DARK.sources[0]));
        assert_eq!(buf[(16, 4)].fg, DARK.color(DARK.sources[1]));
        assert_eq!(buf[(29, 4)].symbol(), "░");
        assert!(read_row(&buf, 5).contains("eremite 52.1 GiB  ■ experts 44.1   ■ rest 8.0"));
        let ssd = read_row(&buf, 6);
        assert!(
            ssd.contains("0.00 / 32.0 GiB   saved states in ~/.cache/eremite/states"),
            "{ssd}"
        );
        let mac = read_row(&buf, 7);
        assert!(
            mac.contains("available 10.7 GiB   pressure normal"),
            "{mac}"
        );
        let normal = find_in_row(&buf, 7, "normal").unwrap();
        assert_eq!(buf[(normal, 7)].fg, DARK.color(DARK.green));
        // ten cells at 100 columns: 42.6 GiB of experts and 7.2 of weights
        // of 107.5 GiB
        let buf = draw(100, 30);
        assert!(read_row(&buf, 4).starts_with(" GPU  █████░░░░░  52.8"));
        assert_eq!(shorten_home("/home/me2/x", "/home/me"), "/home/me2/x");
        assert_eq!(shorten_home("/home/me", "/home/me"), "~");
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

    /// The statuses of two models at a sample `step` of two: deepseek decodes
    /// request 5, and qwen prefills request 6, at layer 30 then 31 of 64,
    /// while request 7 waits.
    fn two_models(step: u64) -> Vec<ModelStatus> {
        let deepseek = format!(
            r#"{{"loaded": true, "phase": "decode",
            "requests": [{{"request": 5, "status": "decoding", "session": 5,
                           "prompt_tokens": 23, "generated_tokens": {}}}]}}"#,
            502 + 14 * step
        );
        let qwen = format!(
            r#"{{"loaded": true, "phase": "prefill",
            "requests": [
                {{"request": 6, "status": "prefill", "session": 1, "prompt_tokens": 12000}},
                {{"request": 7, "status": "waiting", "session": null, "prompt_tokens": 900}}],
            "engine_counters": {{"layers": 64, "prefill_layer": {}, "prefill_rows_now": 12000}}}}"#,
            30 + step
        );
        [deepseek, qwen]
            .map(|json| serde_json::from_str(&json).expect("the worker's form"))
            .to_vec()
    }

    #[test]
    fn the_requests_list_every_models_requests_and_the_log_takes_the_rest() {
        let models = ["deepseek-v4.1-flash", "qwen3.8-27b"].map(|name| ModelInfo {
            name: name.into(),
            kind: Kind::Llm,
        });
        let mut samples = Samples::new(WINDOWS);
        samples.add_sample(0.0, &two_models(0));
        let statuses = two_models(1);
        samples.add_sample(0.5, &statuses);
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 57.0,
            models: &models,
            statuses: &statuses,
            samples: &samples,
            home: "",
            session_text: None,
        };
        let log: Vec<LogLine> = (0..50)
            .map(|i| LogLine {
                time: "19:19:29.080".into(),
                model: Some(i % 2),
                text: format!("line {i}"),
            })
            .collect();
        let mut choices = Choices::default();
        let buf = draw_on_terminal(160, 45, |buf, area| {
            draw_overview(buf, area, &screen, &log, &mut choices);
        });
        assert!(read_row(&buf, 8).starts_with("─ Requests ───"));
        assert!(read_row(&buf, 9).starts_with(
            " request  model                 session  status                      speed         amount"
        ));
        let decoding = read_row(&buf, 10);
        assert!(
            decoding.starts_with(
                " #5       deepseek-v4.1-flash   5        decoding                    28.0 tok/s    516 tokens"
            ),
            "{decoding}"
        );
        // 12,000 rows, 1 of 64 layers further in 0.5 s
        let prefill = read_row(&buf, 11);
        assert!(
            prefill.starts_with(
                " #6       qwen3.8-27b           1        prefill, layer 31 of 64     375 tok/s     12,000 tokens"
            ),
            "{prefill}"
        );
        assert!(read_row(&buf, 12).starts_with(" #7       qwen3.8-27b                    waiting"));
        assert_eq!(buf[(41, 10)].fg, DARK.color(DARK.accent));
        assert_eq!(buf[(41, 11)].fg, DARK.color(DARK.prefill));
        assert_eq!(buf[(41, 12)].fg, DARK.color(DARK.amber));
        // the log in the rows left, its newest line last
        assert!(read_row(&buf, 13).starts_with("─ Log ─ showing all"));
        assert!(read_row(&buf, 44).contains("line 49"));
        assert_eq!(choices.log_page, 31);
        // without requests, and with more than half the rows can take
        let idle = [ModelStatus::default()];
        let screen = Screen {
            statuses: &idle,
            ..screen
        };
        let buf = draw_on_terminal(160, 45, |buf, area| {
            draw_overview(buf, area, &screen, &log, &mut choices);
        });
        assert_eq!(read_row(&buf, 9).trim_end(), " no requests");
        let lines = build_request_lines(&screen, 160, 4);
        assert_eq!(lines.len(), 2, "the rule and the line without requests");
        let crowded = [statuses[0].clone(), statuses[1].clone()];
        let screen = Screen {
            statuses: &crowded,
            ..screen
        };
        let lines = build_request_lines(&screen, 160, 4);
        let text: Vec<String> = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.to_string())
                    .collect()
            })
            .collect();
        assert_eq!(text.len(), 4, "{text:?}");
        assert_eq!(text[3], " … 2 more");
    }

    #[test]
    fn image_requests_show_generating_step_and_image_progress_not_tokens() {
        let models = [ModelInfo {
            name: "minimax-h3-image".into(),
            kind: Kind::Image,
        }];
        let status: ModelStatus = serde_json::from_str(
            r#"{
                "loaded": true,
                "phase": "generating",
                "requests": [{
                    "request": 9,
                    "status": "generating",
                    "session": null,
                    "prompt_tokens": 24,
                    "reused_tokens": 0,
                    "generated_tokens": 0,
                    "images_requested": 4,
                    "generated_images": 2,
                    "routes": []
                }],
                "engine_counters": {
                    "layers": 50,
                    "image_stage": 2,
                    "image_step": 3,
                    "image_steps_total": 6
                }
            }"#,
        )
        .expect("image model status JSON");
        let statuses = [status];
        let mut samples = Samples::new(WINDOWS);
        samples.add_sample(0.0, &statuses);
        samples.add_sample(0.5, &statuses);
        let latest = samples.samples.back().unwrap();
        assert_eq!(latest.models[0].decoding, None);
        assert_eq!(latest.models[0].prefill, None);
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:8008",
            uptime_seconds: 12.0,
            models: &models,
            statuses: &statuses,
            samples: &samples,
            home: "",
            session_text: None,
        };
        let lines = build_request_lines(&screen, 140, 10);
        let row: String = lines[2]
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(row.contains("generating, step 3 of 6"), "{row}");
        assert!(row.contains("2 of 4 images"), "{row}");
        assert!(!row.contains("tok/s"), "{row}");
        assert!(!row.contains("tokens"), "{row}");
    }
}
