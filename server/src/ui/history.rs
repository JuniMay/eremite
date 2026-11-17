//! The view 3 History: the decoding and prefill speeds of all models and of
//! each model, now, recently and since the start.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::Line;

use super::draw::{colored, fit_line, format_count, format_percent, label, plain, strong};
use super::frame::Screen;
use super::samples::{ModelSample, Sample, find_decoding_since_start, find_prefill_since_start};
use super::status::ModelStatus;

/// The columns of a figure's name in the table.
const NAME_COLUMNS: usize = 24;
/// The columns of each figure in the table.
const FIGURE_COLUMNS: usize = 16;
/// The column of the charts' first sample.
const CHART_START: u16 = 16;

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
/// ended requests served.
pub fn draw_history(buf: &mut Buffer, area: Rect, screen: &Screen) {
    let lines = build_table_lines(screen);
    let width = area.width as usize;
    for (row, line) in lines.iter().take(area.height as usize).enumerate() {
        let line = fit_line(screen.theme, line.clone(), width);
        buf.set_line(area.x, area.y + row as u16, &line, area.width);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{Kind, ModelInfo};
    use crate::ui::samples::Samples;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, find_in_row, read_row};
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
}
