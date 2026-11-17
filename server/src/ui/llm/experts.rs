//! An LLM's sub-view Experts: the model's hit rate and read wait, and its
//! decoding requests.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;

use crate::ui::draw::{fit_line, label, plain, strong};
use crate::ui::frame::{Choices, ClickTarget, Screen, Target};
use crate::ui::samples::{find_hit_share, find_read_wait_since_start};
use crate::ui::status::{Activity, ModelStatus, Request};

/// The column of the request tabs.
const TABS_START: u16 = 10;

/// Returns the decoding requests of the model chosen in `choices`, with the
/// one chosen among them: the one that `choices` names, or else the first.
pub fn find_decoding_requests<'a>(
    screen: &'a Screen,
    choices: &Choices,
) -> (Vec<&'a Request>, Option<&'a Request>) {
    let requests: Vec<&Request> = screen
        .statuses
        .get(choices.model)
        .map(|status| {
            status
                .requests
                .iter()
                .filter(|request| request.status == Activity::Decoding)
                .collect()
        })
        .unwrap_or_default();
    let chosen = choices
        .request
        .and_then(|id| requests.iter().copied().find(|request| request.id == id))
        .or(requests.first().copied());
    (requests, chosen)
}

/// Returns whether the model of `status` has no experts to show: it is
/// loaded, and a decoded row chooses no routed experts.
pub fn has_no_experts(status: &ModelStatus) -> bool {
    let counters = &status.engine_counters;
    counters.layers > 0 && counters.experts_per_row == 0
}

/// Draws the Experts of the model chosen in `choices` in `area`: the hit
/// rate and the read wait, and the decoding requests as tabs. The places of
/// the tabs go to `choices`.
pub fn draw_experts(buf: &mut Buffer, area: Rect, screen: &Screen, choices: &mut Choices) {
    let theme = screen.theme;
    let (Some(status), Some(model)) = (
        screen.statuses.get(choices.model),
        screen.models.get(choices.model),
    ) else {
        return;
    };
    if area.height == 0 {
        return;
    }
    if has_no_experts(status) {
        let line = Line::from(plain(format!(" {} has no experts to show.", model.name)));
        buf.set_line(area.x, area.y, &line, area.width);
        return;
    }
    let counters = &status.engine_counters;
    // The model's hit rate and read wait, now, recently and since the start.
    let samples = screen.samples;
    let windows = samples.windows;
    let i = choices.model;
    let hit = |sample: &crate::ui::samples::Sample| sample.models.get(i).and_then(|m| m.hit);
    let wait =
        |sample: &crate::ui::samples::Sample| sample.models.get(i).and_then(|m| m.read_wait_ms);
    let show_hit =
        |value: Option<f64>| value.map_or("—".to_string(), |v| format!("{:.1}%", 100.0 * v));
    let show_wait =
        |value: Option<f64>| value.map_or("—".to_string(), |v| format!("{v:.1} ms/token"));
    let hits = [
        show_hit(samples.find_mean(windows.now, hit)),
        show_hit(samples.find_mean(windows.recent, hit)),
        show_hit(find_hit_share(counters.decode_expert_sources)),
    ];
    let waits = [
        show_wait(samples.find_mean(windows.now, wait)),
        show_wait(samples.find_mean(windows.recent, wait)),
        show_wait(find_read_wait_since_start(status)),
    ];
    let mut figures = Vec::new();
    for (name, values) in [(" hit rate", hits), ("    read wait", waits)] {
        figures.extend([strong(name), plain("   ")]);
        for (value, when) in values
            .into_iter()
            .zip([" now   ", " recent   ", " since start"])
        {
            figures.extend([plain(value), label(theme, when)]);
        }
    }
    let width = area.width as usize;
    buf.set_line(
        area.x,
        area.y,
        &fit_line(theme, Line::from(figures), width),
        area.width,
    );
    let (requests, chosen) = find_decoding_requests(screen, choices);
    let Some(chosen) = chosen else {
        let line = Line::from(vec![plain(" "), label(theme, "no request is decoding")]);
        if area.height > 2 {
            buf.set_line(area.x, area.y + 2, &line, area.width);
        }
        return;
    };
    // The decoding requests as tabs.
    if area.height > 1 {
        let y = area.y + 1;
        buf.set_string(
            area.x + 1,
            y,
            "Request",
            Style::new().fg(theme.color(theme.label)),
        );
        let mut x = area.x + TABS_START;
        for request in &requests {
            let speed = samples
                .find_request_speed(request.id)
                .map_or(String::new(), |speed| format!(" · {speed:.1} tok/s"));
            let session = request
                .session
                .map_or(String::new(), |number| format!(" · session {number}"));
            let text = format!(" #{}{session}{speed} ", request.id);
            let tab_width = text.chars().count() as u16;
            if x + tab_width > area.right() {
                break;
            }
            let style = if request.id == chosen.id {
                Style::new()
                    .fg(theme.color(theme.on_accent))
                    .bg(theme.color(theme.accent))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(theme.color(theme.label))
            };
            buf.set_string(x, y, text, style);
            choices.targets.push(ClickTarget {
                y,
                x0: x,
                x1: x + tab_width,
                target: Target::Request(request.id),
            });
            x += tab_width + 2;
        }
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

    /// The experts that token `token` chose: six in each of 40 layers, and
    /// where each one was. Layer 1's first expert came from the SSD; the
    /// others were in memory.
    fn route(token: usize) -> String {
        let (mut experts, mut sources) = (Vec::new(), Vec::new());
        for layer in 0..40 {
            for j in 0..6 {
                experts.push(((token + 7 * layer + 61 * j) % 384).to_string());
                sources.push(if layer == 0 && j == 0 { "4" } else { "0" });
            }
        }
        format!(
            r#"{{"experts": [{}], "sources": [{}]}}"#,
            experts.join(","),
            sources.join(",")
        )
    }

    /// deepseek-v4.1-flash's status as the worker builds it, while request 5
    /// decodes, with its counters at sample `step` of two.
    fn status(step: u64) -> ModelStatus {
        let routes: Vec<String> = (0..12).map(route).collect();
        let json = format!(
            r#"{{"loaded": true, "phase": "decode",
            "requests": [{{"request": 5, "status": "decoding", "session": 5,
                           "generated_tokens": {}, "routes": [{}]}}],
            "model_decode": {{"computed_rows": {}, "read_wait_seconds": {}}},
            "engine_counters": {{"layers": 43, "experts_per_row": 240, "routed_layers": 40,
                "experts_per_layer": 384, "decode_expert_sources": [{}, 0, 0, 0, {}]}}}}"#,
            100 + 14 * step,
            routes.join(","),
            100 + 14 * step,
            0.1 + 0.014 * step as f64,
            2000 + 900 * step,
            100 * (1 + step)
        );
        serde_json::from_str(&json).expect("the worker's form")
    }

    /// Draws Experts at `width` x `height` from `statuses`, with `choices`.
    fn draw(width: u16, height: u16, statuses: &[ModelStatus], choices: &mut Choices) -> Buffer {
        let models = [ModelInfo {
            name: "deepseek-v4.1-flash".into(),
            kind: Kind::Llm,
        }];
        let mut samples = Samples::new(WINDOWS);
        samples.add_sample(0.0, &[status(0)]);
        samples.add_sample(0.5, &[status(1)]);
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 63.0,
            models: &models,
            statuses,
            samples: &samples,
            home: "",
            session_text: None,
        };
        draw_on_terminal(width, height, |buf, area| {
            draw_experts(buf, area, &screen, choices);
        })
    }

    #[test]
    fn the_figures_and_the_tabs_show_the_models_experts_and_requests() {
        let mut choices = Choices::default();
        let buf = draw(160, 40, &[status(1)], &mut choices);
        // 900 of 1,000 experts in memory now; 14 ms for 14 rows
        let figures = read_row(&buf, 0);
        assert!(
            figures.starts_with(" hit rate   90.0% now   90.0% recent   "),
            "{figures}"
        );
        assert!(
            figures.contains(" read wait   1.0 ms/token now"),
            "{figures}"
        );
        // the decoding request's tab, on the accent
        let tab = find_in_row(&buf, 1, " #5 · session 5 · 28.0 tok/s ").expect("the tab");
        assert_eq!(tab, 10);
        assert_eq!(buf[(tab + 1, 1)].bg, DARK.color(DARK.accent));
        assert_eq!(choices.find_target(tab + 2, 1), Some(Target::Request(5)));
    }

    #[test]
    fn a_model_without_experts_or_requests_says_so() {
        let mut choices = Choices::default();
        let dense: ModelStatus = serde_json::from_str(
            r#"{"loaded": true, "engine_counters": {"layers": 64, "experts_per_row": 0}}"#,
        )
        .unwrap();
        assert!(has_no_experts(&dense));
        assert!(!has_no_experts(&status(1)), "an MoE model");
        assert!(
            !has_no_experts(&ModelStatus::default()),
            "a model not loaded"
        );
        let buf = draw(160, 40, &[dense], &mut choices);
        assert_eq!(
            read_row(&buf, 0).trim_end(),
            " deepseek-v4.1-flash has no experts to show."
        );
        let mut idle = status(1);
        idle.requests.clear();
        let buf = draw(160, 40, &[idle], &mut choices);
        assert_eq!(read_row(&buf, 2).trim_end(), " no request is decoding");
        // tiny terminals
        draw(30, 5, &[status(1)], &mut choices);
        draw(1, 1, &[status(1)], &mut choices);
    }
}
