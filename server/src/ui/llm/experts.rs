//! An LLM's sub-view Experts: the model's hit rate and read wait, and its
//! decoding requests. Under them, the routed experts that a decoding
//! request's latest tokens chose in each layer, each in the color of where it
//! was when it was chosen last. At the right, each layer's experts of those
//! tokens by where each one was.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;

use crate::ui::draw::{LEFT_EIGHTHS, colored, draw_stacked_parts, fit_line, label, plain, strong};
use crate::ui::frame::{Choices, ClickTarget, Screen, Target};
use crate::ui::samples::{find_hit_share, find_read_wait_since_start};
use crate::ui::status::{Activity, ModelStatus, Request};
use crate::ui::theme::Rgb;

/// The cells of each layer's bar.
const BAR_CELLS: u16 = 12;
/// The column of the map's first cell: the layers' numbers take the columns
/// before it.
const MAP_START: u16 = 4;
/// The column of the request tabs.
const TABS_START: u16 = 10;
/// The experts in a cell, the fewest that fit the width first.
const EXPERTS_PER_CELL: [usize; 4] = [2, 4, 8, 16];
/// The experts' numbers along the top of the map, one every this many.
const EXPERTS_PER_MARK: usize = 64;
/// Where an expert was, from the most telling: from the SSD, late,
/// substituted, read ahead, in memory. Of two experts in half a cell, the
/// one chosen last shows, and the most telling one among equals.
const TELLING_SOURCES: [u8; 5] = [4, 2, 3, 1, 0];
/// The names of where an expert was, in the engine's order.
const SOURCE_NAMES: [&str; 5] = ["in memory", "read ahead", "late", "substituted", "from SSD"];
/// The rows above the map: the figures, the request tabs and the experts'
/// numbers.
const ROWS_ABOVE_MAP: u16 = 3;
/// The rows under the map: an empty row, the colors, and an empty row above
/// the keys.
const ROWS_UNDER_MAP: u16 = 3;

/// Returns the columns that the map takes at `per_cell` experts a cell: the
/// layers' numbers, the cells, the scroll bar and the space after it, and the
/// bars.
fn count_map_columns(experts: usize, per_cell: usize) -> usize {
    MAP_START as usize + experts.div_ceil(per_cell) + 2 + BAR_CELLS as usize
}

/// Returns the expert of `group` chosen last: its age in tokens and where it
/// was. The most telling source goes first among equal ages.
fn find_latest(group: &[Option<(usize, u8)>]) -> Option<(usize, u8)> {
    group
        .iter()
        .flatten()
        .copied()
        .min_by_key(|&(age, source)| {
            let telling = TELLING_SOURCES.iter().position(|&t| t == source);
            (age, telling.unwrap_or(TELLING_SOURCES.len()))
        })
}

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
/// rate and the read wait, the decoding requests as tabs, the map of the
/// chosen request's experts with a scroll bar, each layer's bar, and the
/// colors. The places of the tabs go to `choices`.
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
    draw_map(buf, area, screen, choices, chosen);
}

/// Draws the map of `request`'s experts in `area`, under the figures and the
/// tabs: one row per layer with routed experts, the scroll bar, each layer's
/// bar, and the colors. The layers scroll by `choices`.
fn draw_map(
    buf: &mut Buffer,
    area: Rect,
    screen: &Screen,
    choices: &mut Choices,
    request: &Request,
) {
    let theme = screen.theme;
    let Some(status) = screen.statuses.get(choices.model) else {
        return;
    };
    let counters = &status.engine_counters;
    let experts = counters.experts_per_layer as usize;
    let layers = counters.routed_layers as usize;
    let per_layer = (counters.experts_per_row as usize)
        .checked_div(layers)
        .unwrap_or(0);
    let routes = &request.routes;
    let width = area.width as usize;
    let per_cell = EXPERTS_PER_CELL
        .into_iter()
        .find(|&per| count_map_columns(experts, per) <= width);
    let (Some(per_cell), true) = (per_cell, experts > 0 && per_layer > 0) else {
        return;
    };
    if area.height <= ROWS_ABOVE_MAP + ROWS_UNDER_MAP {
        return;
    }
    let label_style = Style::new().fg(theme.color(theme.label));
    let cells = experts.div_ceil(per_cell) as u16;
    let map_x = area.x + MAP_START;
    let map_y = area.y + ROWS_ABOVE_MAP;
    // The experts' numbers, and the last one's at the right.
    let last = (experts - 1).to_string();
    let last_at = cells.saturating_sub(last.len() as u16);
    for expert in (0..experts).step_by(EXPERTS_PER_MARK) {
        let (at, name) = ((expert / per_cell) as u16, expert.to_string());
        if at + (name.len() as u16) < last_at {
            buf.set_string(map_x + at, map_y - 1, name, label_style);
        }
    }
    buf.set_string(map_x + last_at, map_y - 1, last, label_style);
    let bar_x = map_x + cells + 2;
    buf.set_string(
        bar_x,
        map_y - 1,
        format!("{} tokens", routes.len()),
        label_style,
    );
    let rows = (area.height - ROWS_ABOVE_MAP - ROWS_UNDER_MAP) as usize;
    choices.layer_page = rows.max(1);
    choices.layer_scroll = choices.layer_scroll.min(layers.saturating_sub(rows));
    let first_layer = choices.layer_scroll;
    for (row, layer) in (first_layer..layers).take(rows).enumerate() {
        let y = map_y + row as u16;
        let number_columns = MAP_START as usize - 1;
        let number = format!("{:>number_columns$}", layer + 1);
        buf.set_string(area.x, y, number, label_style);
        // Each expert's latest choice in the kept tokens: its age and where
        // it was.
        let mut seen: Vec<Option<(usize, u8)>> = vec![None; experts];
        let mut tally = [0u32; 5];
        for (k, route) in routes.iter().enumerate() {
            let age = routes.len() - 1 - k;
            for j in layer * per_layer..(layer + 1) * per_layer {
                let (Some(&expert), Some(&source)) = (route.experts.get(j), route.sources.get(j))
                else {
                    continue;
                };
                let source = source.min(4);
                if (expert as usize) < experts {
                    seen[expert as usize] = Some((age, source));
                }
                tally[source as usize] += 1;
            }
        }
        for cell in 0..cells as usize {
            let (low, high) = (cell * per_cell, ((cell + 1) * per_cell).min(experts));
            let middle = (low + per_cell / 2).min(high);
            let halves = (
                find_latest(&seen[low..middle]),
                find_latest(&seen[middle..high]),
            );
            let at = &mut buf[(map_x + cell as u16, y)];
            let color = |source: u8| theme.color(theme.sources[source as usize]);
            match halves {
                (Some(left), Some(right)) => {
                    at.set_symbol("▌")
                        .set_fg(color(left.1))
                        .set_bg(color(right.1));
                }
                (Some(left), None) => {
                    at.set_symbol("▌").set_fg(color(left.1));
                }
                (None, Some(right)) => {
                    at.set_symbol("▐").set_fg(color(right.1));
                }
                (None, None) => {}
            }
        }
        // The layer's experts of the kept tokens, by where each one was.
        let total = tally.iter().sum::<u32>().max(1) as f64;
        let parts: Vec<(f64, Rgb)> = (0..5)
            .map(|k| (tally[k] as f64 / total * BAR_CELLS as f64, theme.sources[k]))
            .collect();
        draw_stacked_parts(
            buf,
            theme,
            (bar_x, y),
            (1, 0),
            BAR_CELLS,
            &LEFT_EIGHTHS,
            &parts,
        );
    }
    // The scroll bar: its solid part shows the layers on the screen, of all.
    let shown = rows.min(layers);
    if layers > 0 {
        let thumb = (shown * shown).div_ceil(layers).max(1);
        let thumb_start = first_layer * shown / layers;
        for row in 0..shown {
            let (symbol, rgb) = if (thumb_start..thumb_start + thumb).contains(&row) {
                ("█", theme.rule)
            } else {
                ("│", theme.faint)
            };
            buf[(map_x + cells, map_y + row as u16)]
                .set_symbol(symbol)
                .set_fg(theme.color(rgb));
        }
    }
    // The colors.
    let mut key = vec![plain(" ")];
    for (k, name) in SOURCE_NAMES.iter().enumerate() {
        key.extend([
            colored(theme, theme.sources[k], "■ "),
            plain(format!("{name}   ")),
        ]);
    }
    let key_y = area.bottom() - 2;
    buf.set_line(
        area.x,
        key_y,
        &fit_line(theme, Line::from(key), width),
        area.width,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{Kind, ModelInfo};
    use crate::ui::samples::Samples;
    use crate::ui::status::ModelStatus;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, find_in_row, read_row, read_screen};
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
    fn the_map_shows_each_layers_experts_in_the_colors_of_where_they_were() {
        let mut choices = Choices::default();
        let buf = draw(160, 40, &[status(1)], &mut choices);
        // the experts' numbers, four experts a cell
        assert_eq!(find_in_row(&buf, 2, "64"), Some(4 + 16));
        assert_eq!(find_in_row(&buf, 2, "383"), Some(97));
        assert_eq!(find_in_row(&buf, 2, "12 tokens"), Some(102));
        assert!(read_row(&buf, 3).starts_with("  1 "));
        // layer 1's first expert of the last token: expert 11 from the SSD,
        // in the right half of cell 2
        let cell = &buf[(4 + 2, 3)];
        assert_eq!(cell.symbol(), "▌");
        assert_eq!(cell.bg, DARK.color(DARK.sources[4]));
        // the bars: of layer 1's 72 experts, the 12 from the SSD take the
        // last two of its 12 cells, in red
        assert_eq!(buf[(102, 3)].fg, DARK.color(DARK.sources[0]));
        assert_eq!(buf[(111, 3)].fg, DARK.color(DARK.sources[0]));
        assert_eq!(buf[(112, 3)].fg, DARK.color(DARK.sources[4]));
        assert_eq!(buf[(102, 4)].fg, DARK.color(DARK.sources[0]));
        assert_eq!(
            buf[(113, 4)].fg,
            DARK.color(DARK.sources[0]),
            "layer 2: all in memory"
        );
        // the scroll bar: 34 rows show 34 of the 40 layers, so 29 rows are
        // solid
        assert_eq!(choices.layer_page, 34);
        assert_eq!(buf[(100, 3)].symbol(), "█");
        assert_eq!(buf[(100, 3)].fg, DARK.color(DARK.rule));
        assert_eq!(buf[(100, 3 + 28)].symbol(), "█");
        assert_eq!(buf[(100, 3 + 29)].symbol(), "│");
        assert_eq!(buf[(100, 3 + 29)].fg, DARK.color(DARK.faint));
        let key = read_row(&buf, 38);
        assert!(
            key.starts_with(" ■ in memory   ■ read ahead   ■ late   ■ substituted   ■ from SSD"),
            "{key}"
        );
        // scrolled to the last layers
        choices.layer_scroll = usize::MAX;
        let buf = draw(160, 40, &[status(1)], &mut choices);
        assert_eq!(choices.layer_scroll, 6);
        assert!(
            read_row(&buf, 36).starts_with(" 40 "),
            "{}",
            read_screen(&buf)
        );
        assert_eq!(buf[(100, 36)].symbol(), "█");
        assert_eq!(buf[(100, 3)].symbol(), "│");
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
        // eight experts a cell at 100 columns, and tiny terminals
        let buf = draw(100, 30, &[status(1)], &mut choices);
        assert_eq!(find_in_row(&buf, 2, "64"), Some(4 + 8));
        draw(30, 5, &[status(1)], &mut choices);
        draw(1, 1, &[status(1)], &mut choices);
    }
}
