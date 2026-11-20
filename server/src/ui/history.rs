//! The view 3 History: the decoding and prefill speeds of all models and of
//! each model, now, recently and since the start. Then charts over the
//! history window: each model's speed stacked in its color, where each
//! request arrived, and each request's prefill and decoding.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;

use super::draw::{
    LOWER_EIGHTHS, colored, draw_stacked_parts, fit_line, format_count, format_percent, label,
    plain, strong,
};
use super::frame::Screen;
use super::samples::{
    CHART_COLUMNS, ModelSample, Sample, find_decoding_since_start, find_prefill_since_start,
};
use super::status::{Activity, ModelStatus};
use super::theme::Rgb;

/// The columns of a figure's name in the table.
const NAME_COLUMNS: usize = 24;
/// The columns of each figure in the table.
const FIGURE_COLUMNS: usize = 16;
/// The column of the charts' first sample.
const CHART_START: u16 = 16;
/// The columns at the right of the charts, for what each request does now.
const RIGHT_COLUMNS: u16 = 24;
/// The column of a request's number in its row.
const REQUEST_NUMBER_COLUMN: u16 = 11;
/// The fewest rows of a chart: its top, a middle row and its base.
const LEAST_CHART_ROWS: usize = 3;
/// The time axis names the history window's start and three more quarters.
const AXIS_MARKS: usize = 4;
/// A chart's top is the lowest of these times a power of ten that holds its
/// highest column.
const SCALE_STEPS: [f64; 10] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];

/// A figure of the table: a speed of all models and of each model.
struct Speed {
    /// The figure's name.
    name: &'static str,
    /// A model's speed in a sample, if it has one.
    of: fn(&ModelSample) -> Option<f64>,
    /// The speed since the start of the models given.
    since_start: fn(&[&ModelStatus]) -> Option<f64>,
    /// Shows a speed with its unit.
    show: fn(f64) -> String,
}

/// The two speeds: decoding and prefill, in tokens per second.
const SPEEDS: [Speed; 2] = [
    Speed {
        name: "decoding",
        of: |model| model.decoding,
        since_start: find_decoding_since_start,
        show: |speed| format!("{speed:.1} tok/s"),
    },
    Speed {
        name: "prefill",
        of: |model| model.prefill,
        since_start: find_prefill_since_start,
        show: |speed| format!("{} tok/s", format_count(speed)),
    },
];

/// Draws 3 History in `area`: the colors, the table of speeds and what the
/// ended requests served, then the charts.
pub fn draw_history(buf: &mut Buffer, area: Rect, screen: &Screen) {
    let lines = build_table_lines(screen);
    let width = area.width as usize;
    for (row, line) in lines.iter().take(area.height as usize).enumerate() {
        let line = fit_line(screen.theme, line.clone(), width);
        buf.set_line(area.x, area.y + row as u16, &line, area.width);
    }
    let charts = Rect {
        y: area.y + lines.len() as u16,
        height: area.height.saturating_sub(lines.len() as u16),
        ..area
    };
    draw_charts(buf, charts, screen);
}

/// Builds the lines above the charts: the colors of the prefill, of each
/// model and of an arrival; the speeds now, recently and since the start, of
/// all models and, with several, of each; and what the ended requests served.
fn build_table_lines(screen: &Screen) -> Vec<Line<'static>> {
    let theme = screen.theme;
    let samples = screen.samples;
    let windows = samples.windows;
    let several = screen.models.len() > 1;
    let mut key = vec![
        plain(" ".repeat(CHART_START as usize)),
        colored(theme, theme.prefill, "■"),
        label(theme, " prefill    "),
    ];
    for (i, model) in screen.models.iter().enumerate() {
        key.push(colored(theme, theme.model_color(i), "■"));
        key.push(label(theme, format!(" {}    ", model.name)));
    }
    key.extend([plain("▼"), label(theme, " a request arrived")]);
    let mut head = vec![plain(" ".repeat(1 + NAME_COLUMNS))];
    for name in ["now", "recent", "since start"] {
        head.push(label(theme, format!("{name:<FIGURE_COLUMNS$}")));
    }
    let mut lines = vec![Line::from(key), Line::default(), Line::from(head)];
    let figure = |value: Option<f64>, show: fn(f64) -> String| match value {
        Some(value) => plain(format!("{:<FIGURE_COLUMNS$}", show(value))),
        None => label(theme, format!("{:<FIGURE_COLUMNS$}", "—")),
    };
    let all: Vec<&ModelStatus> = screen.statuses.iter().collect();
    for speed in &SPEEDS {
        let total = |sample: &Sample| sample.sum_models(speed.of);
        let mut line = vec![plain(" "), strong(format!("{:<NAME_COLUMNS$}", speed.name))];
        line.push(figure(samples.find_mean(windows.now, total), speed.show));
        line.push(figure(samples.find_mean(windows.recent, total), speed.show));
        line.push(figure((speed.since_start)(&all), speed.show));
        lines.push(Line::from(line));
        if !several {
            continue;
        }
        for (i, model) in screen.models.iter().enumerate() {
            let of_model = |sample: &Sample| sample.models.get(i).and_then(speed.of);
            // The model's name fits the name's columns, after its square.
            let name_columns = NAME_COLUMNS - 4;
            let mut line = vec![
                plain("   "),
                colored(theme, theme.model_color(i), "■ "),
                plain(format!("{:<name_columns$}", model.name)),
            ];
            line.push(figure(samples.find_mean(windows.now, of_model), speed.show));
            line.push(figure(
                samples.find_mean(windows.recent, of_model),
                speed.show,
            ));
            let since = screen
                .statuses
                .get(i)
                .and_then(|status| (speed.since_start)(&[status]));
            line.push(figure(since, speed.show));
            lines.push(Line::from(line));
        }
    }
    let totals = screen.statuses.iter().map(|status| &status.totals);
    let (mut requests, mut prompt, mut reused, mut generated) = (0, 0, 0, 0);
    for total in totals {
        requests += total.requests;
        prompt += total.prompt_tokens;
        reused += total.reused_tokens;
        generated += total.generated_tokens;
    }
    lines.push(Line::from(vec![
        plain(" "),
        strong(format!("{:<NAME_COLUMNS$}", "served")),
        plain(format!("{} requests", format_count(requests as f64))),
        label(theme, "   prompts "),
        plain(format!("{} tokens", format_count(prompt as f64))),
        label(theme, ", reused "),
        plain(format_percent(reused as f64, prompt as f64)),
        label(theme, "   generated "),
        plain(format!("{} tokens", format_count(generated as f64))),
    ]));
    lines.push(Line::default());
    lines
}

/// Returns the lowest step of SCALE_STEPS times a power of ten that holds
/// `value`.
fn find_chart_top(value: f64) -> f64 {
    let power = 10f64.powf(value.max(f64::MIN_POSITIVE).log10().floor());
    SCALE_STEPS
        .into_iter()
        .map(|step| step * power)
        .find(|&top| top >= value)
        .unwrap_or(SCALE_STEPS[SCALE_STEPS.len() - 1] * power)
}

/// Draws the charts in `area`: decoding and prefill in tokens per second,
/// each model's speed stacked in its color. Under them, where each request
/// arrived, a row for each request in the window, and the time axis. The
/// charts share the rows that the rest leaves.
fn draw_charts(buf: &mut Buffer, area: Rect, screen: &Screen) {
    let theme = screen.theme;
    let samples: Vec<&Sample> = screen.samples.samples.iter().collect();
    let width = (CHART_COLUMNS as u16).min(area.width.saturating_sub(CHART_START + RIGHT_COLUMNS));
    // The samples shown: the newest that the columns hold.
    let first = CHART_COLUMNS - width as usize;
    let sample_at = |column: usize| -> Option<&Sample> {
        let index = (first + column) as i64 - (CHART_COLUMNS as i64 - samples.len() as i64);
        (index >= 0)
            .then(|| samples.get(index as usize).copied())
            .flatten()
    };
    // The requests in the columns shown, in the order they first ran, with
    // their models.
    let mut requests: Vec<(u64, usize)> = Vec::new();
    for sample in (0..width as usize).filter_map(sample_at) {
        for (i, model) in sample.models.iter().enumerate() {
            for &(id, _) in &model.requests {
                if !requests.iter().any(|&(other, _)| other == id) {
                    requests.push((id, i));
                }
            }
        }
    }
    // Each chart takes its name and its base line beside its rows. The
    // arrivals and the time axis take a row each.
    let fixed = 2 * 2 + 2;
    let rows_left = (area.height as usize).saturating_sub(fixed);
    let request_rows = requests
        .len()
        .min(rows_left.saturating_sub(2 * LEAST_CHART_ROWS));
    let chart_rows = (rows_left - request_rows.min(rows_left)) / 2;
    if width == 0 || chart_rows < LEAST_CHART_ROWS {
        if area.height > 0 {
            let note = Line::from(label(theme, " the charts need more rows and columns"));
            buf.set_line(area.x, area.y, &note, area.width);
        }
        return;
    }
    let chart_x = area.x + CHART_START;
    let right = chart_x + width + 2;
    let mut y = area.y;
    for speed in &SPEEDS {
        let highest = samples
            .iter()
            .filter_map(|sample| sample.sum_models(speed.of))
            .fold(1.0, f64::max);
        let top = find_chart_top(highest);
        draw_chart_frame(
            buf,
            screen,
            (area.x, y),
            chart_rows as u16,
            width,
            speed.name,
            top,
        );
        let base = y + chart_rows as u16;
        for column in 0..width as usize {
            let Some(sample) = sample_at(column) else {
                continue;
            };
            let parts: Vec<(f64, Rgb)> = sample
                .models
                .iter()
                .enumerate()
                .filter_map(|(i, model)| {
                    (speed.of)(model)
                        .map(|value| (value / top * chart_rows as f64, theme.model_color(i)))
                })
                .collect();
            let start = (chart_x + column as u16, base);
            draw_stacked_parts(
                buf,
                theme,
                start,
                (0, -1),
                chart_rows as u16,
                &LOWER_EIGHTHS,
                &parts,
            );
        }
        y = base + 2;
    }
    // Where each request arrived.
    for column in 0..width as usize {
        let arrival = sample_at(column).and_then(|sample| {
            sample
                .models
                .iter()
                .flat_map(|model| &model.arrivals)
                .next()
                .copied()
        });
        if let Some(id) = arrival {
            let style = Style::new().fg(theme.color(theme.label));
            buf.set_string(chart_x + column as u16, y, format!("▼#{id}"), style);
        }
    }
    y += 1;
    let latest = samples.last();
    for (row, &(id, i)) in requests.iter().take(request_rows).enumerate() {
        let row_y = y + row as u16;
        if row == 0 {
            buf.set_string(
                area.x + 1,
                row_y,
                "requests",
                Style::new().add_modifier(Modifier::BOLD),
            );
        }
        let tone = theme.model_color(i);
        let style = Style::new()
            .fg(theme.color(tone))
            .add_modifier(Modifier::BOLD);
        buf.set_string(
            area.x + REQUEST_NUMBER_COLUMN,
            row_y,
            format!("#{id}"),
            style,
        );
        let activity_at = |sample: &Sample| {
            sample
                .models
                .get(i)
                .and_then(|model| model.requests.iter().find(|&&(other, _)| other == id))
                .map(|&(_, activity)| activity)
        };
        for column in 0..width as usize {
            let color = match sample_at(column).and_then(activity_at) {
                Some(Activity::Prefill) => theme.prefill,
                Some(Activity::Decoding | Activity::Generating) => tone,
                Some(Activity::Waiting) | None => continue,
            };
            buf[(chart_x + column as u16, row_y)]
                .set_symbol("━")
                .set_fg(theme.color(color));
        }
        let now = match latest.and_then(|sample| activity_at(sample)) {
            Some(Activity::Prefill) => "prefill",
            Some(Activity::Decoding) => "decoding",
            Some(Activity::Generating) => "generating",
            Some(Activity::Waiting) | None => "done",
        };
        buf.set_string(right, row_y, now, Style::new().fg(theme.color(theme.label)));
    }
    draw_time_axis(
        buf,
        screen,
        (chart_x, y + request_rows as u16),
        width,
        first,
    );
}

/// Draws a chart's frame from column `at.0` of row `at.1`: its name above
/// its `rows` rows, its top value and 0 at the left, a dotted rule before its
/// columns, and a base line under them.
fn draw_chart_frame(
    buf: &mut Buffer,
    screen: &Screen,
    at: (u16, u16),
    rows: u16,
    width: u16,
    name: &str,
    top: f64,
) {
    let theme = screen.theme;
    let (x, y) = at;
    let title = Line::from(vec![strong(format!(" {name}")), label(theme, "  tokens/s")]);
    buf.set_line(x, y, &title, CHART_START + width);
    let label_style = Style::new().fg(theme.color(theme.label));
    let scale = |value: String| format!("{value:>14} ");
    buf.set_string(x, y + 1, scale(format!("{top:.0}")), label_style);
    buf.set_string(x, y + rows, scale("0".into()), label_style);
    for row in 1..=rows {
        buf[(x + CHART_START - 1, y + row)]
            .set_symbol("┊")
            .set_fg(theme.color(theme.rule));
    }
    let rule = Style::new().fg(theme.color(theme.rule));
    buf.set_string(
        x + CHART_START,
        y + rows + 1,
        "─".repeat(width as usize),
        rule,
    );
}

/// Draws the time axis from column `at.0` of row `at.1` under `width`
/// columns: the history window's start and three more quarters before now,
/// and "now". `first` is the first sample shown of the window's columns.
fn draw_time_axis(buf: &mut Buffer, screen: &Screen, at: (u16, u16), width: u16, first: usize) {
    let theme = screen.theme;
    let (x, y) = at;
    if y >= buf.area.bottom() {
        return;
    }
    let style = Style::new().fg(theme.color(theme.label));
    let window = screen.samples.windows.history;
    for mark in 0..AXIS_MARKS {
        let column = mark * CHART_COLUMNS / AXIS_MARKS;
        let before = window * (AXIS_MARKS - mark) as f64 / AXIS_MARKS as f64;
        let name = if before.fract() == 0.0 {
            format!("−{before:.0} s")
        } else {
            format!("−{before:.1} s")
        };
        let fits = column >= first && column - first + name.chars().count() + 1 < width as usize;
        if fits {
            buf.set_string(x + (column - first) as u16, y, name, style);
        }
    }
    let now = "now";
    buf.set_string(x + width.saturating_sub(now.len() as u16), y, now, style);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{Kind, ModelInfo};
    use crate::ui::samples::Samples;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, find_in_row, read_row, read_screen};
    use crate::ui::theme::DARK;

    /// A model's status as the worker builds it: its running requests, each
    /// a number, what it does and its generated tokens; the generated tokens
    /// of its ended requests; and its prompt rows and decoding seconds.
    fn status(requests: &[(u64, &str, u64)], ended: u64, rows: u64, seconds: f64) -> ModelStatus {
        let requests: Vec<String> = requests
            .iter()
            .map(|(id, what, generated)| {
                format!(
                    r#"{{"request": {id}, "status": "{what}", "generated_tokens": {generated}}}"#
                )
            })
            .collect();
        let json = format!(
            r#"{{"requests": [{}], "totals": {{"requests": 4, "prompt_tokens": 1938,
                "reused_tokens": 12, "generated_tokens": {ended}, "prefilled_tokens": 1926,
                "prefill_seconds": 24.7}},
                "model_decode": {{"seconds": {seconds}}},
                "engine_counters": {{"layers": 43, "prefill_rows": {rows}}}}}"#,
            requests.join(", ")
        );
        serde_json::from_str(&json).expect("the worker's form")
    }

    /// 130 samples, half a second apart, of two models. deepseek-v4.1-flash
    /// prefills request 1 from sample 10 and decodes it from 20 to 40.
    /// Request 2 prefills from sample 50 and decodes from 55 on. qwen3.8-27b
    /// prefills request 3 from sample 100 on.
    fn samples() -> (Samples, Vec<ModelStatus>) {
        let mut samples = Samples::new(WINDOWS);
        let mut statuses = Vec::new();
        let (mut deepseek_tokens, mut deepseek_rows, mut qwen_rows) = (0, 0, 0);
        for k in 0..130u64 {
            let mut deepseek = Vec::new();
            match k {
                10..20 => deepseek.push((1, "prefill", 0)),
                20..40 => deepseek.push((1, "decoding", 10 * (k - 20))),
                50..55 => deepseek.push((2, "prefill", 0)),
                55.. => deepseek.push((2, "decoding", 10 * (k - 55))),
                _ => {}
            }
            let ended = if k >= 40 { 200 } else { 0 };
            deepseek_rows += if matches!(k, 10..20 | 50..55) { 50 } else { 0 };
            if matches!(k, 20..40 | 55..) {
                deepseek_tokens += 1;
            }
            let qwen: Vec<(u64, &str, u64)> = if k >= 100 {
                qwen_rows += 200;
                vec![(3, "prefill", 0)]
            } else {
                Vec::new()
            };
            statuses = vec![
                status(
                    &deepseek,
                    ended,
                    deepseek_rows,
                    0.5 * deepseek_tokens as f64,
                ),
                status(&qwen, 0, qwen_rows, 0.0),
            ];
            samples.add_sample(0.5 * k as f64, &statuses);
        }
        (samples, statuses)
    }

    /// Draws 3 History at `width` x `height`.
    fn draw(width: u16, height: u16) -> Buffer {
        let models = ["deepseek-v4.1-flash", "qwen3.8-27b"].map(|name| ModelInfo {
            name: name.into(),
            kind: Kind::Llm,
        });
        let (samples, statuses) = samples();
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 65.0,
            models: &models,
            statuses: &statuses,
            samples: &samples,
            home: "",
            session_text: None,
        };
        draw_on_terminal(width, height, |buf, area| draw_history(buf, area, &screen))
    }

    #[test]
    fn the_table_shows_the_speeds_of_all_models_and_of_each() {
        let buf = draw(160, 41);
        let key = read_row(&buf, 0);
        assert!(
            key.starts_with(
                "                ■ prefill    ■ deepseek-v4.1-flash    ■ qwen3.8-27b    ▼ a request arrived"
            ),
            "{key}"
        );
        assert_eq!(buf[(29, 0)].fg, DARK.color(DARK.model_color(0)));
        assert_eq!(buf[(54, 0)].fg, DARK.color(DARK.model_color(1)));
        assert_eq!(find_in_row(&buf, 2, "now"), Some(25));
        // 10 tokens a sample, two samples a second; 600 tokens since the start
        // in their 37.5 s
        assert!(read_row(&buf, 3).starts_with(" decoding                20.0 tok/s"));
        let deepseek = read_row(&buf, 4);
        assert!(
            deepseek.starts_with("   ■ deepseek-v4.1-flash 20.0 tok/s"),
            "{deepseek}"
        );
        assert_eq!(buf[(3, 4)].fg, DARK.color(DARK.model_color(0)));
        assert!(read_row(&buf, 5).starts_with("   ■ qwen3.8-27b         —"));
        // qwen's 200 rows a sample now
        assert!(read_row(&buf, 6).starts_with(" prefill                 400 tok/s"));
        assert!(read_row(&buf, 9).starts_with(
            " served                  8 requests   prompts 3,876 tokens, reused 0.6%   generated 200 tokens"
        ));
    }

    #[test]
    fn the_charts_stack_each_models_speed_in_its_color() {
        let buf = draw(160, 41);
        // the decoding chart's frame and scale: 20 tokens a second at most
        assert!(read_row(&buf, 11).starts_with(" decoding  tokens/s"));
        assert!(read_row(&buf, 12).starts_with("            20 ┊"));
        assert_eq!(buf[(15, 12)].fg, DARK.color(DARK.rule));
        let deepseek = DARK.color(DARK.model_color(0));
        let qwen = DARK.color(DARK.model_color(1));
        // request 1 decodes in the window's columns 10 to 29, in deepseek's
        // color
        assert_eq!(buf[(16 + 15, 21)].symbol(), "█");
        assert_eq!(buf[(16 + 15, 21)].fg, deepseek);
        assert_eq!(buf[(16 + 5, 21)].symbol(), " ");
        assert_eq!(
            read_row(&buf, 22),
            format!("{}{}{}", " ".repeat(16), "─".repeat(120), " ".repeat(24))
        );
        // qwen prefills at the end, in its color, under a scale of 400
        assert!(read_row(&buf, 24).starts_with("           400 ┊"));
        assert_eq!(buf[(16 + 119, 33)].fg, qwen);
        // where each request arrived
        let arrivals = read_row(&buf, 35);
        assert_eq!(find_in_row(&buf, 35, "▼#2"), Some(16 + 40), "{arrivals}");
        assert_eq!(find_in_row(&buf, 35, "▼#3"), Some(16 + 90), "{arrivals}");
        // each request's row: its prefill, then its decoding in its model's
        // color, and what it does now
        assert!(read_row(&buf, 36).starts_with(" requests  #1   ━"));
        assert_eq!(buf[(11, 36)].fg, deepseek);
        assert_eq!(buf[(16, 36)].fg, DARK.color(DARK.prefill));
        assert_eq!(buf[(16 + 15, 36)].fg, deepseek);
        assert_eq!(find_in_row(&buf, 36, "done"), Some(138));
        assert_eq!(find_in_row(&buf, 37, "decoding"), Some(138));
        assert_eq!(find_in_row(&buf, 38, "prefill"), Some(138));
        assert_eq!(buf[(11, 38)].fg, qwen);
        let axis = read_row(&buf, 39);
        assert!(
            axis.starts_with("                −60 s                         −45 s"),
            "{axis}"
        );
        assert_eq!(find_in_row(&buf, 39, "now"), Some(133));
    }

    #[test]
    fn the_charts_fit_narrow_and_tiny_terminals() {
        // 60 columns: the newest half of the window
        let buf = draw(100, 26);
        // the table's 11 rows, two charts of 3 rows, the arrivals, two
        // requests, then the axis
        let axis = read_row(&buf, 24);
        assert!(
            axis.starts_with("                −30 s                         −15 s"),
            "{axis}"
        );
        assert!(
            !read_screen(&buf).contains("#1 "),
            "request 1 is out of the columns shown"
        );
        let buf = draw(100, 14);
        assert!(read_row(&buf, 11).starts_with(" the charts need more rows and columns"));
        draw(1, 1);
        assert_eq!(find_chart_top(28.1), 30.0);
        assert_eq!(find_chart_top(452.0), 500.0);
        assert_eq!(find_chart_top(0.5), 0.5);
        assert_eq!(find_chart_top(1.0), 1.0);
    }
}
